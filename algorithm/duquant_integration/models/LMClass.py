import transformers
import torch
import time
from algorithm import utils
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

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
        print("max_gen_toks fn")
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
            # Reset per-forward activation accumulator if available
            try:
                if hasattr(self, '_act_stats') and self._act_stats is not None:
                    utils.reset_activation_current(self._act_stats)
            except Exception:
                pass

            num_tokens = int(inps.numel())
            use_cuda = torch.cuda.is_available() and (
                (isinstance(self.device, torch.device) and self.device.type == 'cuda')
                or (isinstance(self.device, str) and str(self.device).startswith('cuda'))
            )

            try:
                if use_cuda:
                    try:
                        torch.cuda.reset_peak_memory_stats(self.device)
                    except Exception:
                        pass
                    baseline_alloc = torch.cuda.memory_allocated(self.device)
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = self.model(inps)["logits"]
                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                elapsed = t1 - t0
                if use_cuda:
                    try:
                        peak_alloc = torch.cuda.max_memory_allocated(self.device)
                        gpu_activation_peak = int(max(0, peak_alloc - baseline_alloc))
                    except Exception:
                        gpu_activation_peak = 0
                else:
                    gpu_activation_peak = 0
            except Exception:
                t0 = time.perf_counter()
                logits = self.model(inps)["logits"]
                t1 = time.perf_counter()
                elapsed = t1 - t0
                gpu_activation_peak = 0

            # Record throughput to base LM counters
            try:
                self.record_model_call(num_tokens, elapsed)
            except Exception:
                pass

            # Aggregate GPU activation peak on the LM object
            try:
                self._act_gpu_peak_max = max(getattr(self, '_act_gpu_peak_max', 0), int(gpu_activation_peak))
            except Exception:
                pass

            return logits

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
