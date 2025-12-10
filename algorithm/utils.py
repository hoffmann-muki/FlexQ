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
# from torch._six import inf
from math import inf
import logging
from termcolor import colored
import sys
import os
import time

def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # ensure output directory exists (accept Path objects too)
    try:
        out_dir_str = str(output_dir)
    except Exception:
        out_dir_str = output_dir
    if out_dir_str and not os.path.exists(out_dir_str):
        os.makedirs(out_dir_str, exist_ok=True)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(out_dir_str, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def _tensor_bytes(x):
    """Return size in bytes for a tensor or a nested structure of tensors."""
    if x is None:
        return 0
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(v) for v in x)
    if isinstance(x, dict):
        return sum(_tensor_bytes(v) for v in x.values())
    return 0


def attach_activation_memory_hooks(model, modules_filter=None):
    """Attach forward hooks that accumulate activation output sizes.

    Returns (handles, stats) where handles is a list of hook handles (callable.remove()),
    and stats is a dict with keys:
      - 'peak_bytes': max accumulated bytes observed across forwards
      - 'per_layer_peak': dict of per-module peak output bytes
      - 'current_bytes': current forward accumulation (resets per-forward by caller)

    modules_filter: optional callable(name, module) -> bool to select modules to hook.
    If None, hooks are attached to all submodules.
    """
    stats = {
        'peak_bytes': 0,
        'per_layer_peak': {},
        'current_bytes': 0,
    }
    handles = []

    for name, module in model.named_modules():
        if modules_filter is not None and not modules_filter(name, module):
            continue

        def _make_hook(n):
            def _hook(mod, inp, out):
                size = _tensor_bytes(out)
                stats['current_bytes'] += size
                prev = stats['per_layer_peak'].get(n, 0)
                if size > prev:
                    stats['per_layer_peak'][n] = size
                if stats['current_bytes'] > stats['peak_bytes']:
                    stats['peak_bytes'] = stats['current_bytes']

            return _hook

        handle = module.register_forward_hook(_make_hook(name))
        handles.append(handle)

    return handles, stats


def reset_activation_current(stats):
    """Reset the per-forward accumulation counter (call before each forward)."""
    stats['current_bytes'] = 0


def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass