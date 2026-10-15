#!/usr/bin/env python3
"""Separates quantization error from engine bugs: top-1 agreement with the bfloat16 reference answers
(crates/tachyon/tests/fixtures/gemma-4-e4b.ref) of Hugging Face's Gemma 4 E4B with its weights rounded as Tachyon
rounds them, and with alternatives. Each variant reads every prompt and reference answer in one pass and picks the
best token after each answer position, as the engine test does. If Tachyon's own figure matches "q4 g64 /7 + q8",
the engine computes what its weights allow.

With `vision.ref` (`scripts/media.py vision`), each answer is about an image: the float32 tower's soft tokens fill
its slots, as they did for the reference, and only the decoder's weights are rounded. With `speech.ref`
(`scripts/media.py speech`), each answer is a transcription: the audio tower runs in float32, as it did for the
reference.

    python3 scripts/agreement.py ~/models/gemma-4-E4B-it crates/tachyon/tests/fixtures/gemma-4-e4b.ref
    python3 scripts/agreement.py ~/models/gemma-4-E4B-it crates/tachyon/tests/fixtures/media/vision.ref
    python3 scripts/agreement.py ~/models/gemma-4-E4B-it crates/tachyon/tests/fixtures/media/speech.ref
"""

import json
import os
import sys

import torch
from transformers import Gemma4ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def soft_tokens(src, lines):
    """Each `vision.ref` line's float32 soft tokens, as `scripts/media.py vision` made them."""
    from PIL import Image, ImageOps
    from transformers import AutoProcessor
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    import media
    proc = AutoProcessor.from_pretrained(src)
    model = Gemma4ForConditionalGeneration.from_pretrained(src, dtype=torch.float32).cuda().eval()
    out = []
    for j in lines:
        img = ImageOps.exif_transpose(Image.open(os.path.join(media.ROOT, j["image"]))).convert("RGB")
        proc.image_processor = Gemma4ImageProcessorPil(max_soft_tokens=j["budget"])
        enc = proc(text=[j["prompt"]], images=[[img]], return_tensors="pt").to("cuda")
        assert enc["input_ids"][0].tolist() == j["ids"], j["image"]
        with torch.no_grad():
            out.append(model.model.get_image_features(enc["pixel_values"], enc["image_position_ids"]).pooler_output[0].float().cpu())
    del model
    torch.cuda.empty_cache()
    return out


def features(src, lines):
    """Each `speech.ref` line's clip as the processor gives it to the model: features and their mask."""
    from transformers import AutoProcessor
    import media
    proc = AutoProcessor.from_pretrained(src)
    out = []
    for j in lines:
        enc = proc(text=[j["prompt"]], audio=[media.clip(os.path.join(media.ROOT, j["clip"]))], sampling_rate=16000, return_tensors="pt")
        assert enc["input_ids"][0].tolist() == j["ids"], j["clip"]
        out.append({k: enc[k].to("cuda") for k in ("input_features", "input_features_mask")})
    return out


def agreement(model, lines, soft):
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    agree = total = 0
    for k, j in enumerate(lines):
        prompt = j.get("ids", j["prompt"])
        ids = torch.tensor([prompt + j["answer"]], device="cuda")
        extra = soft[k] if soft and "clip" in j else {}
        if soft and "image" in j:
            extra["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(soft[k].to("cuda", torch.bfloat16),))}
        with torch.no_grad():
            logits = model(input_ids=ids, **extra).logits[0]
        n = len(prompt)
        picks = logits[n - 1 : n - 1 + len(j["answer"])].argmax(-1).tolist()
        agree += sum(p == a for p, a in zip(picks, j["answer"]))
        total += len(j["answer"])
    return agree, total


def main(src, ref):
    lines = [json.loads(l) for l in open(ref)]
    soft = soft_tokens(src, lines) if "image" in lines[0] else features(src, lines) if "clip" in lines[0] else None
    variants = [
        ("bf16", None, False),
        ("q4 g64 /7", (64, [7]), False),
        ("q4 g64 /7 + q8", (64, [7]), True),
        ("q4 g64 best scale", (64, [6 + i / 4 for i in range(13)]), False),
        ("q4 g64 best scale + q8 (Tachyon)", (64, [6 + i / 4 for i in range(13)]), True),
        ("q4 g32 /7", (32, [7]), False),
        ("q4 g32 best scale", (32, [6 + i / 4 for i in range(13)]), False),
    ]
    for name, quant, acts in variants:
        model = Gemma4ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16).to("cuda").eval()
        model.model.audio_tower.float()
        model.model.embed_audio.float()
        lm = model.model.language_model
        with torch.no_grad():
            for mod in [m for m in lm.modules() if isinstance(m, (torch.nn.Linear, torch.nn.Embedding))] if quant else []:
                mod.weight.copy_(q4(mod.weight, *quant))
        if acts:
            for mod in [m for m in lm.modules() if isinstance(m, torch.nn.Linear)] + [model.lm_head]:
                mod.register_forward_pre_hook(lambda _, a: (q8(a[0]),) + a[1:])
        agree, total = agreement(model, lines, soft)
        print(f"{name:<28} {agree}/{total} ({100 * agree / total:.2f}%)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
