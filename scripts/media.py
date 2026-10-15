#!/usr/bin/env python3
"""Step 11's media references.

`python3 scripts/media.py images` writes crates/tachyon/tests/fixtures/media/: PNG files of every color type, bit
depth and row filter (encoded here, row by row) and JPEG files of every chroma subsampling PIL writes, grayscale,
restart markers, Adobe RGB, EXIF orientations, odd sizes, plus a progressive and an interlaced file Tachyon refuses.
Then `images.ref`, one JSON line per file: the RGB that PIL decodes (`Image.open(…)`, `exif_transpose`,
`convert("RGB")`, as `transformers` loads an image) as size and FNV-1a 64 hash, and for each soft-token budget the
size Gemma 4's image processor (`Gemma4ImageProcessorPil`) resizes it to and the hash of PIL's bicubic result. Needs
only Pillow and numpy.

`python3 scripts/media.py vision ~/models/gemma-4-E4B-it` is the vision tower's precision study and references (a
GPU with 40 GB, torch and transformers 5). For three fixture images at 70 and 280 soft tokens it runs the tower and
its projection into the decoder in float32 (TF32 off) and again with every linear layer's weight and input rounded to
bf16 and to f16, and prints how far the soft tokens move and the largest value any linear layer reads or writes
(f16 ends at 65504). Then it writes crates/tachyon/tests/fixtures/media/vision.ref: per image and budget the prompt's
tokens through the chat template, the float32 soft tokens' file (`soft-<image>-<budget>.f16`, `f16 [n, 2560]`, for
the 70-token runs) and the decoder's greedy answer in bfloat16 to those soft tokens, with the margin of each token.

`python3 scripts/media.py audio [~/models/gemma-4-E4B-it]` writes the audio fixtures: WAV files of every sample
format Tachyon reads (8-bit unsigned to 64-bit float, plain and `WAVE_FORMAT_EXTENSIBLE`, mono and stereo, an extra
chunk before the data) and `audio.ref`, one JSON line per file: its rate, its samples and the FNV-1a 64 of them as
`f32` (`soundfile`'s scaling, `v / 2^(bits-1)`, 8 bits `(v - 128) / 128`; channels averaged as `librosa.to_mono`
does), and for the 16 kHz files the log-mel features of Gemma 4's feature extractor (`mel-<name>.f32`, `f32 [frames,
128]`). With a checkpoint, its feature extractor's settings are checked against the defaults Tachyon implements.
Needs numpy and transformers.

`python3 scripts/media.py speech ~/models/gemma-4-E4B-it clip.wav...` is the audio tower's precision study and
references (as `vision`'s), on speech clips of 16 kHz mono 16-bit WAV (up to 30 s; `ffmpeg -i in -ac 1 -ar 16000
-c:a pcm_s16le out.wav` makes one), copied into the fixtures. It runs the audio tower and its projection in float32
and with every linear and convolution layer's weight and input rounded to bf16 and to f16, printing how far the soft
tokens move and the largest values layers read, write and leave in the residual stream. Then `speech.ref`: per clip
the prompt's tokens (the question, then the audio), the float32 soft tokens' file (`soft-<clip>.f16`), and the
decoder's greedy answer in bfloat16 with the float32 tower, with each token's margin.

`python3 scripts/media.py tiny-audio` writes crates/tachyon/tests/fixtures/tiny-audio/: a Gemma 4 audio tower of
random weights (hidden 128, one head, two layers, clip bounds that bite; bfloat16 safetensors under Gemma 4's names,
with its embedder into 256 dimensions), its `config.json`, and `soft.f32`, its float32 soft tokens of the fixture
`speech-like.wav` through `transformers`' feature extractor, tower and embedder: the GPU tower's test with no
checkpoint. Needs torch on the CPU.

`python3 scripts/media.py inspect ~/models/gemma-4-E4B-it` prints what converting the vision tower needs to know: its
tensors (layer 0's, as every layer has the same), whether the clip bounds of linear layers reading the same input
agree (so they can share one product), and the pad token's embedding row.
"""

import io
import json
import math
import os
import struct
import sys
import zlib

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "tachyon", "tests", "fixtures", "media")
BUDGETS = (70, 280, 1120)


def fnv(data):
    """FNV-1a 64 of `data`."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def picture(w, h, seed):
    """An RGB picture with gradients, stripes, a disc, sharp edges and noise: something for every codec path."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 127 + 120 * np.sin(x / (3 + seed % 5)) * np.cos(y / 7)
    g = 255 * x / max(w - 1, 1)
    b = 255 * y / max(h - 1, 1)
    disc = (x - w / 2) ** 2 + (y - h / 2) ** 2 < (min(w, h) / 3) ** 2
    img = np.stack([r, g, b], -1)
    img[disc] = [250, 40, 10]
    img[(x.astype(int) // 8 + y.astype(int) // 8) % 5 == 0] = [0, 0, 0]
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def png(w, h, color, depth, rows, level=6, interlace=0, palette=None, trns=None):
    """A PNG of raw `rows` (bytes per row, unfiltered), each row filtered by type `y % 5`."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    bpp = max(1, channels * depth // 8)
    out, prev = b"", bytes(len(rows[0]))
    for y, row in enumerate(rows):
        f = y % 5
        filtered = bytearray()
        for i, v in enumerate(row):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = [0, a, b, (a + b) // 2, a if pa <= pb and pa <= pc else (b if pb <= pc else c)][f]
            filtered.append((v - pred) & 255)
        out += bytes([f]) + bytes(filtered)
        prev = row
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, interlace))
    if palette is not None:
        data += chunk(b"PLTE", bytes(palette))
    if trns is not None:
        data += chunk(b"tRNS", bytes(trns))
    data += chunk(b"tEXt", b"Comment\x00made by scripts/media.py")
    z = zlib.compress(out, level)
    return data + chunk(b"IDAT", z[: len(z) // 2]) + chunk(b"IDAT", z[len(z) // 2:]) + chunk(b"IEND", b"")


def pack(values, depth):
    """Samples of `depth` bits packed into bytes, most significant first."""
    if depth == 8:
        return bytes(int(v) for v in values)
    if depth == 16:
        return b"".join(struct.pack(">H", int(v)) for v in values)
    out, acc, n = bytearray(), 0, 0
    for v in values:
        acc, n = acc << depth | int(v), n + depth
        if n == 8:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def pngs():
    """Every PNG color type and bit depth, stored and compressed, with every row filter."""
    files = {}
    pic = picture(37, 23, 1)
    gray = pic.mean(-1)
    for depth in (1, 2, 4, 8, 16):
        top = (1 << depth) - 1
        rows = [pack(np.round(gray[y] / 255 * top), depth) for y in range(23)]
        files["gray%d.png" % depth] = png(37, 23, 0, depth, rows)
    for depth in (8, 16):
        k = 256 if depth == 16 else 1
        rows = [pack(pic[y].reshape(-1).astype(np.int64) * k + (y % 3 if depth == 16 else 0), depth) for y in range(23)]
        files["rgb%d.png" % depth] = png(37, 23, 2, depth, rows, level=0 if depth == 8 else 9)
        ga = np.stack([gray, 255 - gray], -1)
        rows = [pack(ga[y].reshape(-1).astype(np.int64) * k, depth) for y in range(23)]
        files["graya%d.png" % depth] = png(37, 23, 4, depth, rows)
        rgba = np.concatenate([pic, gray[..., None].astype(np.uint8)], -1)
        rows = [pack(rgba[y].reshape(-1).astype(np.int64) * k, depth) for y in range(23)]
        files["rgba%d.png" % depth] = png(37, 23, 6, depth, rows)
    for depth in (1, 2, 4, 8):
        n = 1 << depth
        palette = [int(v) for i in range(n - 1) for v in picture(1, 1, 100 + i)[0, 0]]
        index = (gray * (n - 1) / 255).astype(int)  # the last index has no entry: black
        rows = [pack(index[y], depth) for y in range(23)]
        files["palette%d.png" % depth] = png(37, 23, 3, depth, rows, palette=palette, trns=[128] * 3)
    big = picture(400, 300, 2)
    files["big.png"] = png(400, 300, 2, 8, [big[y].tobytes() for y in range(300)], level=1)
    files["interlaced.png"] = png(8, 8, 2, 8, [bytes(24)] * 8, interlace=1)
    return files


def jpeg(arr, **kw):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "JPEG", **kw)
    return b.getvalue()


def jpegs():
    """PIL's JPEGs: subsamplings, qualities, odd sizes, grayscale, restarts, Adobe RGB, EXIF turns."""
    files = {}
    for (w, h), seed in (((101, 67), 3), ((64, 48), 4), ((17, 9), 5), ((3, 2), 6)):
        pic = picture(w, h, seed)
        for sub in (0, 1, 2):
            for q in (30, 90):
                files["s%d-q%d-%dx%d.jpg" % (sub, q, w, h)] = jpeg(pic, quality=q, subsampling=sub)
    pic = picture(101, 67, 7)
    files["gray.jpg"] = jpeg(pic.mean(-1).astype(np.uint8), quality=80)
    files["optimized.jpg"] = jpeg(pic, quality=75, optimize=True)
    files["restart.jpg"] = jpeg(pic, quality=75, restart_marker_blocks=3)
    files["restart-rows.jpg"] = jpeg(pic, quality=75, subsampling=2, restart_marker_rows=1)
    files["adobe-rgb.jpg"] = jpeg(pic, quality=85, keep_rgb=True)
    files["q100.jpg"] = jpeg(pic, quality=100, subsampling=0)
    for o in (3, 6, 8):
        exif = Image.Exif()
        exif[0x0112] = o
        files["exif%d.jpg" % o] = jpeg(pic, quality=80, exif=exif.tobytes())
    files["photo.jpg"] = jpeg(picture(1200, 900, 8), quality=85)
    files["progressive.jpg"] = jpeg(pic, quality=75, progressive=True)
    return files


def fitted(w, h, budget, patch=16, pool=3):
    """Gemma4ImageProcessorPil's target size (its `get_aspect_ratio_preserving_size`), as (width, height)."""
    max_patches = budget * pool * pool
    factor = math.sqrt(max_patches * patch ** 2 / (h * w))
    side = pool * patch
    th, tw = int(math.floor(factor * h / side)) * side, int(math.floor(factor * w / side)) * side
    most = (max_patches // pool ** 2) * side
    if th == 0:
        th, tw = side, min(int(math.floor(w / h)) * side, most)
    elif tw == 0:
        tw, th = side, min(int(math.floor(h / w)) * side, most)
    return tw, th


def images():
    os.makedirs(ROOT, exist_ok=True)
    files = {**pngs(), **jpegs()}
    with open(os.path.join(ROOT, "images.ref"), "w") as ref:
        for name, data in sorted(files.items()):
            with open(os.path.join(ROOT, name), "wb") as f:
                f.write(data)
            line = {"file": name}
            try:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                img.load()
            except Exception as e:  # noqa: BLE001 — PIL refuses what Tachyon refuses
                line["error"] = type(e).__name__
                ref.write(json.dumps(line) + "\n")
                continue
            if name in ("interlaced.png", "progressive.jpg"):
                line["refused"] = True
            if name == "gray16.png":
                line["clipped"] = True  # PIL clips 16-bit gray to 255; Tachyon takes the high byte, as for RGB
            line.update(width=img.width, height=img.height, fnv=fnv(img.tobytes()))
            sizes = {}
            for budget in BUDGETS:
                tw, th = fitted(img.width, img.height, budget)
                out = img if (tw, th) == img.size else img.resize((tw, th), Image.BICUBIC)
                sizes[str(budget)] = [tw, th, fnv(out.tobytes())]
            line["fitted"] = sizes
            ref.write(json.dumps(line) + "\n")
    print("%d files in %s" % (len(files), ROOT))


# ------------------------------------------------------------------------------------------------ vision study
STUDY = ("photo.jpg", "big.png", "exif6.jpg")
QUESTION = "Describe this image in one sentence."
ANSWER = 32


class Rounded:
    """Within the block, every linear and convolution layer under `modules` sees its weight and its input rounded to
    `dtype`."""

    def __init__(self, modules, dtype):
        import torch
        kinds = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
        self.linears = [m for mod in modules for m in mod.modules() if isinstance(m, kinds)]
        self.dtype, self.saved, self.hooks = dtype, [], []

    def round(self, x):
        return x.to(self.dtype).float()

    def __enter__(self):
        for m in self.linears:
            self.saved.append(m.weight.data)
            m.weight.data = self.round(m.weight.data)
            self.hooks.append(m.register_forward_pre_hook(lambda _, args: (self.round(args[0]),) + args[1:]))
        return self

    def __exit__(self, *_):
        for m, w in zip(self.linears, self.saved):
            m.weight.data = w
        for h in self.hooks:
            h.remove()


def extremes(modules):
    """Hooks recording the largest |value| any linear or convolution layer under `modules` reads and writes, and any
    audio layer leaves in the residual stream."""
    import torch
    seen = {"in": 0.0, "out": 0.0}

    def hook(_, args, out):
        seen["in"] = max(seen["in"], float(args[0].abs().max()))
        seen["out"] = max(seen["out"], float(out.abs().max()))
    def layer(_, args, out):
        seen["stream"] = max(seen.get("stream", 0.0), float(out.abs().max()))
    kinds = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
    hooks = [m.register_forward_hook(hook) for mod in modules for m in mod.modules() if isinstance(m, kinds)]
    hooks += [m.register_forward_hook(layer) for mod in modules for m in mod.modules() if type(m).__name__.endswith("AudioLayer")]
    return seen, hooks


def vision(checkpoint):
    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    proc = AutoProcessor.from_pretrained(checkpoint)
    model = Gemma4ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32).cuda().eval()
    tower = [model.model.vision_tower, model.model.embed_vision]
    bounds = [b for m in model.model.vision_tower.modules() for n, b in m.named_buffers(recurse=False) if n.startswith(("input_", "output_"))]
    print("clip bounds: %d, finite: %d, tightest %.3g" % (len(bounds), sum(bool(torch.isfinite(b)) for b in bounds),
                                                         min((float(b.abs()) for b in bounds if torch.isfinite(b)), default=float("inf"))))
    runs = []
    for name in STUDY:
        img = ImageOps.exif_transpose(Image.open(os.path.join(ROOT, name))).convert("RGB")
        for budget in (70, 280):
            proc.image_processor = Gemma4ImageProcessorPil(max_soft_tokens=budget)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            enc = proc(text=[text], images=[[img]], return_tensors="pt").to("cuda")
            runs.append((name, budget, text, enc))
    print("\n%-10s %7s %-6s %12s %12s %12s %10s %10s" % ("image", "budget", "format", "max |d|", "max rel", "min cosine", "max in", "max out"))
    refs = []
    for name, budget, _, enc in runs:
        with torch.no_grad():
            seen, hooks = extremes(tower)
            ref = model.model.get_image_features(enc["pixel_values"], enc["image_position_ids"]).pooler_output[0].float()
            for h in hooks:
                h.remove()
            refs.append(ref)
            print("%-10s %7d %-6s %12s %12s %12s %10.3g %10.3g" % (name, budget, "f32", "", "", "", seen["in"], seen["out"]))
            for label, dtype in (("bf16", torch.bfloat16), ("f16", torch.float16)):
                with Rounded(tower, dtype):
                    got = model.model.get_image_features(enc["pixel_values"], enc["image_position_ids"]).pooler_output[0].float()
                d = (got - ref).abs()
                cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
                print("%-10s %7d %-6s %12.3e %12.3e %12.6f %10s %10s" % (
                    name, budget, label, float(d.max()), float(d.max() / ref.abs().max()), float(cos.min()), "", "") +
                    ("  NON-FINITE" if not torch.isfinite(got).all() else ""))
    model.to(torch.bfloat16)
    with open(os.path.join(ROOT, "vision.ref"), "w") as out:
        for (name, budget, text, enc), ref in zip(runs, refs):
            line = {"image": name, "budget": budget, "prompt": text, "ids": enc["input_ids"][0].tolist(), "soft_tokens": ref.shape[0]}
            if budget == 70:
                line["soft"] = "soft-%s-%d.f16" % (name.split(".")[0], budget)
                ref.half().cpu().numpy().tofile(os.path.join(ROOT, line["soft"]))
            line["answer"], line["margins"] = greedy(model, enc["input_ids"], ref)
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            print("%s at %d: %s" % (name, budget, proc.decode(line["answer"])))


def greedy(model, ids, soft):
    """The decoder's greedy answer in bfloat16 to `ids` with `soft` (float32 soft tokens) in its image slots, and
    the gap between the best logit and the runner-up at each step."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    feats = {"image": BaseModelOutputWithPooling(pooler_output=(soft.to(torch.bfloat16),))}
    answer, margins = [], []
    with torch.no_grad():
        out = model(input_ids=ids, mm_encoder_outputs=feats, use_cache=True)
        for _ in range(ANSWER):
            top = out.logits[0, -1].float().topk(2)
            token = int(top.indices[0])
            answer.append(token)
            margins.append(round(float(top.values[0] - top.values[1]), 4))
            if token in (1, 106):
                break
            out = model(input_ids=torch.tensor([[token]], device=ids.device), past_key_values=out.past_key_values, use_cache=True)
    return answer, margins


def inspect(checkpoint):
    """What the vision tower's conversion needs to know of the checkpoint: its tensors' names, shapes and types,
    whether the clip bounds of linear layers that read the same input agree, and the pad token's embedding row."""
    import glob
    import torch
    from safetensors import safe_open
    tensors = {}
    for path in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if "vision" in k or k.endswith("embed_tokens.weight"):
                    tensors[k] = f.get_tensor(k)
    for k, t in sorted(tensors.items()):
        if "layers." in k and "layers.0." not in k:
            continue
        print("%-90s %-14s %s" % (k, str(t.dtype).replace("torch.", ""), list(t.shape) if t.dim() else float(t)))
    bound = lambda k: float(tensors[k]) if k in tensors else None
    layers = sorted({int(k.split("layers.")[1].split(".")[0]) for k in tensors if "layers." in k})
    for group in (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj")):
        same = 0
        for i in layers:
            pre = [k for k in tensors if k.endswith("layers.%d.%s.input_min" % (i, group[0]))][0][: -len(group[0] + ".input_min")]
            ins = [(bound(pre + g + ".input_min"), bound(pre + g + ".input_max")) for g in group]
            same += len(set(ins)) == 1
            if i in (0, layers[-1]):
                print("layer %d %s input bounds: %s" % (i, "/".join(g.split(".")[-1] for g in group), ins))
        print("%s: input bounds equal in %d of %d layers" % ("/".join(g.split(".")[-1] for g in group), same, len(layers)))
    emb = [t for k, t in tensors.items() if k.endswith("language_model.embed_tokens.weight") or k == "model.embed_tokens.weight"]
    if emb:
        print("pad (token 0) embedding row: max |v| %.6g, norm %.6g" % (float(emb[0][0].abs().max()), float(emb[0][0].float().norm())))


def sound(n, rate, seed):
    """`n` samples at `rate` of tones, a chirp and noise, peaking at 0.9: something for every mel bin."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 2500 * t + 1) + 0.2 * np.sin(2 * np.pi * (100 + 3000 * t) * t)
    x += rng.normal(0, 0.05, n) * (1 + np.sin(2 * np.pi * 3 * t))
    return 0.9 * x / np.abs(x).max()


def wav(path, x, rate, tag, bits, extensible=False, extra=False):
    """Writes `x` (`[n, channels]` in [-1, 1]) as a WAV file of format `tag` (1 integers, 3 floats) and `bits`; returns
    the samples as Tachyon reads them, `f32`, channels averaged."""
    full = 2 ** (bits - 1)
    if tag == 3:
        data = x.astype("<f4" if bits == 32 else "<f8")
        read = data.astype(np.float32)
    elif bits == 8:
        data = np.clip(np.round(x * 128) + 128, 0, 255).astype(np.uint8)
        read = (data.astype(np.float32) - 128) / 128
    else:
        q = np.clip(np.round(x * full), -full, full - 1).astype(np.int64)
        read = q.astype(np.float32) / np.float32(full)
        data = q.astype("<i4") if bits == 32 else q.astype("<i2") if bits == 16 else q
    raw = data.tobytes() if bits != 24 else b"".join(int(v).to_bytes(3, "little", signed=True) for v in data.reshape(-1))
    channels = x.shape[1]
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    if extensible:
        fmt += struct.pack("<HHI", 22, bits, 3) + struct.pack("<H", tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if extra:
        chunks += b"LIST" + struct.pack("<I", 5) + b"INFOx\x00"
    chunks += b"data" + struct.pack("<I", len(raw)) + raw + b"\x00" * (len(raw) % 2)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)
    return read.sum(axis=1, dtype=np.float32) / np.float32(channels) if channels > 1 else read[:, 0]


def audio(checkpoint=None):
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    ours = Gemma4AudioFeatureExtractor()
    if checkpoint:
        from transformers import AutoFeatureExtractor
        theirs = AutoFeatureExtractor.from_pretrained(checkpoint)
        for k, v in vars(ours).items():
            if not isinstance(v, np.ndarray) and getattr(theirs, k, None) != v:
                print("checkpoint's feature extractor differs: %s = %r, Tachyon's %r" % (k, getattr(theirs, k, None), v))
        same = all(np.array_equal(getattr(ours, k), getattr(theirs, k)) for k in ("window", "mel_filters"))
        print("window and mel filters " + ("the same" if same else "DIFFER"))
    mono = sound(4000, 16000, 1)[:, None]
    stereo = np.stack([mono[:, 0], sound(4000, 16000, 2)], 1)
    files = [
        ("audio-u8.wav", mono, 16000, 1, 8, {}),
        ("audio-s16.wav", mono, 16000, 1, 16, {}),
        ("audio-s24.wav", mono, 16000, 1, 24, {}),
        ("audio-s32.wav", mono, 16000, 1, 32, {}),
        ("audio-f32.wav", mono, 16000, 3, 32, {}),
        ("audio-f64.wav", mono, 16000, 3, 64, {}),
        ("audio-stereo.wav", stereo, 16000, 1, 16, {}),
        ("audio-extensible.wav", stereo, 16000, 1, 24, {"extensible": True, "extra": True}),
        ("audio-44100.wav", sound(11025, 44100, 3)[:, None], 44100, 1, 16, {}),
        ("speech-like.wav", sound(19745, 16000, 4)[:, None], 16000, 1, 16, {}),
    ]
    with open(os.path.join(ROOT, "audio.ref"), "w") as out:
        for name, x, rate, tag, bits, kw in files:
            read = wav(os.path.join(ROOT, name), x, rate, tag, bits, **kw)
            line = {"file": name, "rate": rate, "samples": len(read), "fnv": fnv(read.astype("<f4").tobytes())}
            if rate == 16000 and name in ("audio-s16.wav", "speech-like.wav", "audio-stereo.wav"):
                feats = ours([read], return_tensors="np")["input_features"][0]
                line["mel"] = "mel-%s.f32" % name.split(".")[0]
                line["frames"] = feats.shape[0]
                feats.astype("<f4").tofile(os.path.join(ROOT, line["mel"]))
            out.write(json.dumps(line) + "\n")
            print(line)


# ------------------------------------------------------------------------------------------------ audio study
TRANSCRIBE = "Transcribe this audio."


def clip(path):
    """A 16 kHz mono 16-bit WAV file's samples, as Tachyon reads them."""
    import wave
    with wave.open(path) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path + ": not 16 kHz mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / np.float32(32768)


def speech(checkpoint, paths):
    import shutil
    import torch
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    proc = AutoProcessor.from_pretrained(checkpoint)
    model = Gemma4ForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32).cuda().eval()
    tower = [model.model.audio_tower, model.model.embed_audio]
    bounds = [b for m in model.model.audio_tower.modules() for n, b in m.named_buffers(recurse=False) if n.startswith(("input_", "output_"))]
    print("clip bounds: %d, finite: %d" % (len(bounds), sum(bool(torch.isfinite(b)) for b in bounds)))
    runs = []
    for path in paths:
        name = os.path.basename(path)
        shutil.copyfile(path, os.path.join(ROOT, name))
        msgs = [{"role": "user", "content": [{"type": "text", "text": TRANSCRIBE}, {"type": "audio"}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = proc(text=[text], audio=[clip(path)], sampling_rate=16000, return_tensors="pt").to("cuda")
        runs.append((name, text, enc))

    def soft(enc):
        out = model.model.get_audio_features(enc["input_features"], enc["input_features_mask"])
        return out.pooler_output[out.attention_mask].float()
    print("\n%-24s %-6s %12s %12s %12s %10s %10s %10s" % ("clip", "format", "max |d|", "max rel", "min cosine", "max in", "max out", "stream"))
    refs = []
    for name, _, enc in runs:
        with torch.no_grad():
            seen, hooks = extremes(tower)
            ref = soft(enc)
            for h in hooks:
                h.remove()
            refs.append(ref)
            print("%-24s %-6s %12s %12s %12s %10.3g %10.3g %10.3g" % (name, "f32", "", "", "", seen["in"], seen["out"], seen.get("stream", 0)))
            for label, dtype in (("bf16", torch.bfloat16), ("f16", torch.float16)):
                with Rounded(tower, dtype):
                    got = soft(enc)
                d = (got - ref).abs()
                cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
                print("%-24s %-6s %12.3e %12.3e %12.6f" % (name, label, float(d.max()), float(d.max() / ref.abs().max()), float(cos.min())) +
                      ("  NON-FINITE" if not torch.isfinite(got).all() else ""))
    model.to(torch.bfloat16)
    for m in tower:
        m.float()
    with open(os.path.join(ROOT, "speech.ref"), "w") as out:
        for (name, text, enc), ref in zip(runs, refs):
            line = {"clip": name, "prompt": text, "ids": enc["input_ids"][0].tolist(), "soft_tokens": ref.shape[0], "frames": enc["input_features"].shape[1]}
            line["soft"] = "soft-%s.f16" % name.split(".")[0]
            ref.half().cpu().numpy().tofile(os.path.join(ROOT, line["soft"]))
            line["answer"], line["margins"] = transcribe(model, enc)
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            print("%s: %s" % (name, proc.decode(line["answer"])))


def transcribe(model, enc):
    """The decoder's greedy answer in bfloat16 (the audio tower in float32), with each token's margin."""
    import torch
    answer, margins = [], []
    with torch.no_grad():
        out = model(**enc, use_cache=True)
        for _ in range(4 * ANSWER):
            top = out.logits[0, -1].float().topk(2)
            token = int(top.indices[0])
            answer.append(token)
            margins.append(round(float(top.values[0] - top.values[1]), 4))
            if token in (1, 106):
                break
            out = model(input_ids=torch.tensor([[token]], device="cuda"), past_key_values=out.past_key_values, use_cache=True)
    return answer, margins


def tiny_audio():
    import torch
    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
    torch.manual_seed(11)
    cfg = Gemma4AudioConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=1, output_proj_dims=256)

    class Text:
        hidden_size = 256
    tower, embed = Gemma4AudioModel(cfg).eval(), Gemma4MultimodalEmbedder(cfg, Text()).eval()
    with torch.no_grad():
        for mod in (tower, embed):
            for n, p in mod.named_parameters():
                norm = "norm" in n and n.endswith("weight")
                p.copy_((1.0 + 0.2 * torch.randn_like(p)) if norm else torch.randn_like(p) * (0.5 if p.dim() == 1 else 1.5 / p.shape[-1] ** 0.5))
                p.copy_(p.to(torch.bfloat16).float())
            for n, b in mod.named_buffers():
                if n.endswith(("_min", "_max")):
                    b.fill_((-1 if n.endswith("_min") else 1) * float(torch.empty(()).uniform_(2.0, 6.0)))
        # Queries, keys and values read the same input, under the same bounds, as in Gemma 4's checkpoints.
        for layer in tower.layers:
            for p in (layer.self_attn.k_proj, layer.self_attn.v_proj):
                p.input_min.copy_(layer.self_attn.q_proj.input_min)
                p.input_max.copy_(layer.self_attn.q_proj.input_max)
    dst = os.path.join(ROOT, "..", "tiny-audio")
    os.makedirs(dst, exist_ok=True)
    tensors = {"model.audio_tower." + k: v.contiguous() for k, v in tower.state_dict().items() if not k.endswith("inv_timescales")}
    tensors.update({"model.embed_audio." + k: v.contiguous() for k, v in embed.state_dict().items()})
    tensors = {k: v.to(torch.bfloat16) if v.dim() > 0 else v.float() for k, v in tensors.items()}
    save_file(tensors, os.path.join(dst, "model.safetensors"))
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump({"audio_config": cfg.to_dict()}, f, indent=1, default=str)
    with wave_open(os.path.join(ROOT, "speech-like.wav")) as samples:
        feats = Gemma4AudioFeatureExtractor()([samples], return_tensors="pt")
    with torch.no_grad():
        out = tower(feats["input_features"], feats["input_features_mask"])
        soft = embed(out.last_hidden_state[out.attention_mask])
    soft.float().numpy().astype("<f4").tofile(os.path.join(dst, "soft.f32"))
    print("%d soft tokens of %d, largest %.3g, in %s" % (soft.shape[0], soft.shape[1], float(soft.abs().max()), dst))


class wave_open:
    """A 16 kHz mono 16-bit WAV file's samples, as Tachyon reads them, in a `with`."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return clip(self.path)

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "images"
    if mode == "vision":
        vision(os.path.expanduser(sys.argv[2]))
    elif mode == "inspect":
        inspect(os.path.expanduser(sys.argv[2]))
    elif mode == "speech":
        speech(os.path.expanduser(sys.argv[2]), [os.path.expanduser(p) for p in sys.argv[3:]])
    elif mode == "tiny-audio":
        tiny_audio()
    elif mode == "audio":
        audio(os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        images()
