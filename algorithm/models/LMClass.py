import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb
import logging

logger = logging.getLogger(__name__)

class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        # choose device based on provided device_map to avoid unintended moves
        device_map = getattr(args, "device_map", None)
        if isinstance(device_map, str):
            if device_map.lower().startswith("cuda"):
                self._device = torch.device(device_map)
            elif device_map.lower() == "cpu":
                self._device = torch.device("cpu")
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = getattr(args, "batch_size", 1)

        self.model_config = args.model

        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        # Load model with memory-efficient options: device_map='auto' places layers on available devices
        # and low_cpu_mem_usage=True reduces peak CPU memory during loading.
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        device_map = getattr(args, "device_map", "auto")
        low_cpu = getattr(args, "low_cpu_mem_usage", True)
        dtype = getattr(args, "torch_dtype", torch.float16)
        if isinstance(dtype, str):
            if dtype == "auto":
                dtype = "auto"
            else:
                dtype = getattr(torch, dtype, torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu,
            torch_dtype=dtype,
        )
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        logger.info("vocab size: %s", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        logger.debug("max_gen_toks called")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )