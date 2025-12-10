# Copyright 2024 ByteDance and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from algorithm.models.int_llama_layer import QuantLlamaDecoderLayer, QuantLlamaAttention
from algorithm.models.int_opt_layer import QuantOPTDecoderLayer, QuantOPTAttention
from algorithm.flexq_quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
from algorithm import utils
import os
import pdb
from torch.nn import functional as F
import gc
import logging
import numpy as np
from algorithm.flexq_quantize.utils import let_parameters, lwc_parameters, get_abq_parameters,com_parameters, \
                            abq_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            weight_quant_inplace,clear_temp_variable,set_quant_state


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     


def apply_layer_quant_schedule(qlayer, schedule_entry, layer_idx: int, logger=None):
    if not schedule_entry:
        return
    mlp = getattr(qlayer, 'mlp', None)
    if mlp is None or not hasattr(mlp, 'down_proj'):
        return
    quantizer = getattr(mlp.down_proj, 'act_quantizer', None)
    if quantizer is None:
        return

    clip_pct = schedule_entry.get('clip_percentile')
    if clip_pct is not None:
        quantizer.clip_percentile = clip_pct
    zero_shift = schedule_entry.get('zero_shift_scale')
    if zero_shift is not None:
        quantizer.zero_shift_scale = zero_shift
    if logger is not None and hasattr(logger, 'debug'):
        clip_str = f"{clip_pct:.4f}" if clip_pct is not None else "None"
        zero_str = f"{zero_shift:.4f}" if zero_shift is not None else "None"
        severity = schedule_entry.get('severity') or 1.0
        logger.debug(
            f"Layer {layer_idx} schedule: clip_pct={clip_str} zero_shift={zero_str} severity={severity:.2f}"
        )

def flexqllm(
    lm,
    args,
    logger=None,
):
    logger.info("Starting ...")

    # move embedding layer and first layer to target device
    model = lm.model
    # `load_dev` is where the HF model was loaded (usually CPU). `quant_dev`
    # is where we temporarily move a single layer for quantization (GPU if available).
    load_dev = lm.device
    quant_dev = torch.device("cuda") if torch.cuda.is_available() else load_dev
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False

    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        # keep the model loaded on `load_dev` (typically CPU)
        model.model.embed_tokens = model.model.embed_tokens.to(load_dev)
        model.model.norm = model.model.norm.to(load_dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            "down_proj":"fc2"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(load_dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(load_dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(load_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(load_dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1",
            "fc2":"fc2"
        }
        layer_name_prefix = "model.decoder.layers"
    else:
        raise ValueError("Only support for llama/Llama-2 now")
    
    
    layers[0] = layers[0].to(load_dev)

    dtype = torch.float16
    traincast = torch.cuda.amp.autocast

    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        # If this layer was already processed by DuQuant, skip re-quantizing it with FlexQ.
        orig_layer = layers[i]
        if getattr(orig_layer, '_duquant_processed', False):
            logger.info(f"Skipping layer {i} (DuQuant processed)")
            # ensure the layer lives on the load device for later evaluation
            try:
                layers[i] = orig_layer.to(load_dev)
            except Exception:
                layers[i] = orig_layer.to("cpu")
            if quant_dev.type == "cuda":
                torch.cuda.empty_cache()
            continue
        # Move only this layer to `quant_dev` for quantization to avoid OOM
        layer = orig_layer.to(quant_dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        schedule_entry = None
        if hasattr(args, 'layer_clip_schedule'):
            schedule_entry = args.layer_clip_schedule.get(i)
        apply_layer_quant_schedule(qlayer, schedule_entry, i, logger)
        qlayer = qlayer.to(quant_dev)

        set_quant_state(qlayer, weight_quant=True, act_quant=True)
            
        weight_quant_inplace(qlayer, args, is_llama)

        qlayer.half()
        # Register quantization flags into the model
        register_scales_and_zeros(qlayer)
        # move quantized layer back to the load device (usually CPU)
        try:
            layers[i] = qlayer.to(load_dev)
        except Exception:
            # fallback to CPU
            layers[i] = qlayer.to("cpu")

        del layer
        # clear CUDA cache only if we used CUDA
        if quant_dev.type == "cuda":
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

