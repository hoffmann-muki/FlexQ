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


def get_hellaswag_prompts(nsamples, seed, seqlen, model):
    print("get_hellaswag")
    dataset = load_dataset("hellaswag", split="validation")
    tokenizer, pad_id = _default_tokenizer(model)

    def format_fn(doc):
        # Try common hellaswag field layouts: 'endings' (list) or 'ending0'..'ending3'
        choices = []
        if isinstance(doc.get('endings', None), (list, tuple)):
            choices = doc.get('endings')
        else:
            for i in range(4):
                k = f'ending{i}'
                if k in doc:
                    choices.append(doc[k])

        # Fallback to a generic 'choices' field if present
        if not choices and 'choices' in doc:
            c = doc.get('choices')
            if isinstance(c, (list, tuple)):
                choices = c

        # Context field variants
        context = doc.get('ctx') or doc.get('context') or doc.get('article') or ''

        # Build a readable prompt
        choices_text = ' / '.join(choices) if choices else ' / '.join(
            [str(v) for k, v in doc.items() if isinstance(v, str) and k.lower().startswith('ending')][:4]
        )
        return f"Context: {context}\nChoices: {choices_text}\nAnswer:"

    prompts = _get_prompt_texts(dataset, format_fn)
    loader = _build_prompt_loader(prompts, tokenizer, nsamples, seed, seqlen, pad_id)
    return loader, None


def get_winogrande_prompts(nsamples, seed, seqlen, model):
    """Build prompt loader for Winogrande-style tasks.

    Handles multiple possible HF dataset field layouts:
      - 'options' (list of choices)
      - 'option0'..'optionN'
      - 'sentence' with placeholder and 'answer'
      - 'sentence1'/'sentence2' style
    """
    print("get_winogrande")
    # winogrande requires a config name; try a set of sensible defaults and
    # fall back to the first available config if needed.
    configs_to_try = [
        "winogrande_debiased",
        "winogrande_xl",
        "winogrande_m",
        "winogrande_s",
        "winogrande_l",
        "winogrande_xs",
    ]
    dataset = None
    last_exc = None
    for cfg in configs_to_try:
        try:
            dataset = load_dataset("winogrande", cfg, split="validation")
            break
        except Exception as e:
            last_exc = e
            continue
    if dataset is None:
        # Give the original error back if none of the defaults worked
        raise last_exc
    tokenizer, pad_id = _default_tokenizer(model)

    def format_fn(doc):
        # Try 'options' list first
        choices = []
        if isinstance(doc.get('options', None), (list, tuple)):
            choices = doc.get('options')

        # Try option0, option1, ... pattern
        if not choices:
            for i in range(6):
                k = f'option{i}'
                if k in doc:
                    choices.append(doc[k])

        # Try common two-sentence layout (sentence1/sentence2)
        sentence = doc.get('sentence') or doc.get('sent') or ''
        if not sentence:
            s1 = doc.get('sentence1') or doc.get('sent1')
            s2 = doc.get('sentence2') or doc.get('sent2')
            if s1 and s2:
                sentence = f"{s1} {s2}"

        # If still no choices, search for fields that look like endings/choices
        if not choices:
            choices = [v for k, v in doc.items() if isinstance(v, str) and k.lower().startswith('ending')]

        # Fallback to any short string fields (not ideal but robust)
        if not choices:
            candidates = [v for k, v in doc.items() if isinstance(v, str) and len(v) < 200]
            # pick up to 4 distinct small fields
            seen = set()
            for v in candidates:
                if v not in seen:
                    choices.append(v)
                    seen.add(v)
                if len(choices) >= 4:
                    break

        choices_text = ' / '.join(choices) if choices else ''
        return f"Context: {sentence}\nChoices: {choices_text}\nAnswer:"

    prompts = _get_prompt_texts(dataset, format_fn)
    loader = _build_prompt_loader(prompts, tokenizer, nsamples, seed, seqlen, pad_id)
    return loader, None


def get_boolq_prompts(nsamples, seed, seqlen, model):
    print("get_boolq")
    dataset = load_dataset("boolq", split="validation")
    tokenizer, pad_id = _default_tokenizer(model)
    def format_fn(doc):
        text = doc["passage"]
        question = doc["question"]
        return f"Passage: {text}\nQuestion: {question}\nAnswer:"
    prompts = _get_prompt_texts(dataset, format_fn)
    loader = _build_prompt_loader(prompts, tokenizer, nsamples, seed, seqlen, pad_id)
    return loader, None


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model="") -> Tuple:
    lower = name.lower()
    if "wikitext2" in lower:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "piqa" in lower:
        return get_piqa_prompts(nsamples, seed, seqlen, model)
    if "boolq" in lower:
        return get_boolq_prompts(nsamples, seed, seqlen, model)
    if "arc_easy" in lower:
        return get_arc_prompts(nsamples, seed, seqlen, model, "ARC-Easy")
    if "arc_challenge" in lower:
        return get_arc_prompts(nsamples, seed, seqlen, model, "ARC-Challenge")
    if "hellaswag" in lower:
        return get_hellaswag_prompts(nsamples, seed, seqlen, model)
    if "winogrande" in lower:
        return get_winogrande_prompts(nsamples, seed, seqlen, model)
    raise ValueError(f"Unsupported dataset '{name}'")