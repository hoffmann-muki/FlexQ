import random
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def _pad_or_truncate(ids: torch.Tensor, seqlen: int, pad_id: int) -> torch.Tensor:
    if ids.shape[0] >= seqlen:
        return ids[:seqlen]
    pad_len = seqlen - ids.shape[0]
    pad_tensor = torch.full((pad_len,), pad_id, dtype=torch.long)
    return torch.cat([ids, pad_tensor], dim=0)


def _default_tokenizer(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or tokenizer.cls_token_id or 0
    return tokenizer, pad_id


def _build_prompt_loader(prompts: List[str], tokenizer, nsamples, seed, seqlen, pad_id):
    random.seed(seed)
    loader = []
    prompt_count = len(prompts)
    if prompt_count == 0:
        raise RuntimeError("No prompts were loaded for calibration")
    for _ in range(nsamples):
        doc = prompts[random.randrange(prompt_count)]
        enc = tokenizer(doc, return_tensors="pt", truncation=True, max_length=seqlen)
        ids = enc.input_ids.squeeze(0)
        ids = _pad_or_truncate(ids, seqlen, pad_id)
        loader.append({"input_ids": ids.unsqueeze(0)})
    return loader


def get_wikitext2(nsamples, seed, seqlen, model):
    print("get_wikitext2")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer, _ = _default_tokenizer(model)

    def _collate_text(rows):
        return tokenizer("\n\n".join(rows), return_tensors="pt")

    train_enc = _collate_text(train["text"])
    test_enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    total = train_enc.input_ids.shape[1]
    loader = []
    random.seed(seed)
    for _ in range(nsamples):
        if total <= seqlen + 1:
            idx = 0
        else:
            idx = random.randint(0, total - seqlen - 1)
        chunk = train_enc.input_ids[:, idx : idx + seqlen]
        target = chunk.clone()
        target[:, :-1] = -100
        loader.append((chunk, target))
    return loader, test_enc


def _get_prompt_texts(dataset, format_fn):
    return [format_fn(doc) for doc in dataset]


def _format_arc_prompt(doc):
    choices = doc.get("choices", {}).get("text", [])
    choices_text = " / ".join(choices)
    return f"Question: {doc['question']}\nChoices: {choices_text}\nAnswer:"


def get_arc_prompts(nsamples, seed, seqlen, model, variant):
    print(f"get_arc_{variant.lower()}")
    dataset = load_dataset("ai2_arc", variant, split="validation")
    tokenizer, pad_id = _default_tokenizer(model)
    prompts = _get_prompt_texts(dataset, _format_arc_prompt)
    loader = _build_prompt_loader(prompts, tokenizer, nsamples, seed, seqlen, pad_id)
    return loader, None


def get_piqa_prompts(nsamples, seed, seqlen, model):
    print("get_piqa")
    dataset = load_dataset("piqa", split="validation")
    tokenizer, pad_id = _default_tokenizer(model)
    prompts = _get_prompt_texts(dataset, lambda doc: f"Question: {doc['goal']}\nAnswer:")
    loader = _build_prompt_loader(prompts, tokenizer, nsamples, seed, seqlen, pad_id)
    return loader, None


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model="") -> Tuple:
    lower = name.lower()
    if "wikitext2" in lower:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "piqa" in lower:
        return get_piqa_prompts(nsamples, seed, seqlen, model)
    if "arc_easy" in lower:
        return get_arc_prompts(nsamples, seed, seqlen, model, "ARC-Easy")
    if "arc_challenge" in lower:
        return get_arc_prompts(nsamples, seed, seqlen, model, "ARC-Challenge")
    raise ValueError(f"Unsupported dataset '{name}'")