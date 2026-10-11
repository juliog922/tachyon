#!/usr/bin/env python3
"""Separates quantization error from engine bugs: top-1 agreement with the bfloat16 reference answers
(crates/tachyon/tests/fixtures/gemma-4-e4b.ref) of Hugging Face's Gemma 4 E4B with its weights rounded as Tachyon
rounds them, and with alternatives. Each variant reads every prompt and reference answer in one pass and picks the
best token after each answer position, as the engine test does. If Tachyon's own figure matches "q4 g64 /7 + q8",
the engine computes what its weights allow.

    python3 scripts/agreement.py ~/models/gemma-4-E4B-it crates/tachyon/tests/fixtures/gemma-4-e4b.ref
"""

import json
import sys

import torch
from transformers import Gemma4ForConditionalGeneration


def q4(w, group, divisors):
    """`w` rounded to 4 bits in groups of `group`: per group, the f16 scale amax/k (k from `divisors`) whose
    rounding to -8..7 errs least."""
    out = torch.empty_like(w)
    for i in range(0, w.shape[0], 4096):
        x = w[i : i + 4096].float().reshape(-1, group)
        amax = x.abs().amax(1, keepdim=True)
        best, err = None, None
        for k in divisors:
            d = (amax / k).half().float()
            y = (x / torch.where(d == 0, 1, d)).round().clamp(-8, 7) * d
            e = (y - x).square().sum(1, keepdim=True)
            best, err = (y, e) if best is None else (torch.where(e < err, y, best), torch.minimum(e, err))
        out[i : i + 4096] = best.reshape(-1, w.shape[1]).to(w.dtype)
    return out


def q8(x):
    """Activations rounded to 8 bits in blocks of 32, as the kernels round them."""
    b = x.float().reshape(-1, 32)
    s = b.abs().amax(1, keepdim=True) / 127
    return ((b / torch.where(s == 0, 1, s)).round() * s).reshape(x.shape).to(x.dtype)


def agreement(model, lines):
    agree = total = 0
    for j in lines:
        ids = torch.tensor([j["prompt"] + j["answer"]], device="cuda")
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]
        n = len(j["prompt"])
        picks = logits[n - 1 : n - 1 + len(j["answer"])].argmax(-1).tolist()
        agree += sum(p == a for p, a in zip(picks, j["answer"]))
        total += len(j["answer"])
    return agree, total


def main(src, ref):
    lines = [json.loads(l) for l in open(ref)]
    variants = [
        ("bf16", None, False),
        ("q4 g64 /7 (Tachyon)", (64, [7]), False),
        ("q4 g64 /7 + q8 (Tachyon)", (64, [7]), True),
        ("q4 g64 best scale", (64, [6 + i / 4 for i in range(13)]), False),
        ("q4 g32 /7", (32, [7]), False),
        ("q4 g32 best scale", (32, [6 + i / 4 for i in range(13)]), False),
    ]
    for name, quant, acts in variants:
        model = Gemma4ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16).to("cuda").eval()
        lm = model.model.language_model
        with torch.no_grad():
            for mod in [m for m in lm.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Embedding))] if quant else []:
                mod.weight.copy_(q4(mod.weight, *quant))
        if acts:
            for mod in [m for m in lm.modules() if isinstance(m, torch.nn.Linear)] + [model.lm_head]:
                mod.register_forward_pre_hook(lambda _, a: (q8(a[0]),) + a[1:])
        agree, total = agreement(model, lines)
        print(f"{name:<28} {agree}/{total} ({100 * agree / total:.2f}%)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
