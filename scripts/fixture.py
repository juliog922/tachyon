#!/usr/bin/env python3
"""Tokenizer fixtures: the strings the tokenizer tests encode, and the reference token IDs for them.

    scripts/fixture.py strings                  writes crates/tachyon/tests/fixtures/strings.jsonl (once)
    scripts/fixture.py toy                      trains a small Gemma-shaped tokenizer on them:
                                                crates/tachyon/tests/fixtures/toy.json
    scripts/fixture.py ids TOKENIZER.json NAME  encodes every string with Hugging Face `tokenizers`, the reference,
                                                into crates/tachyon/tests/fixtures/NAME.ids
    scripts/fixture.py chat TOKENIZER.json TEMPLATE.jinja NAME
                                                renders sample conversations with the model's own chat template,
                                                as Hugging Face does, into crates/tachyon/tests/fixtures/NAME.chat

An .ids file has one line per string: the number of tokens and the FNV-1a 64 hash of their IDs as little-endian
u32s. A failing test names the string; `scripts/fixture.py show TOKENIZER.json INDEX` prints its reference IDs.

Needs `pip install tokenizers faker jinja2`. Not part of the build: the outputs are committed.
"""

import json
import random
import sys
import sysconfig
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "crates/tachyon/tests/fixtures"
STRINGS = FIXTURES / "strings.jsonl"
COUNT = 10_000
LOCALES = """ar_AA bg_BG bn_BD cs_CZ da_DK de_DE el_GR en_US es_ES fa_IR fi_FI fr_FR he_IL hi_IN hu_HU hy_AM id_ID
it_IT ja_JP ka_GE ko_KR nl_NL pl_PL pt_BR ro_RO ru_RU sv_SE ta_IN th_TH tr_TR uk_UA vi_VN zh_CN zh_TW""".split()
EMOJI = ["😀", "👍🏽", "👨‍👩‍👧‍👦", "🇪🇸", "🏳️‍🌈", "❤️", "🤖", "🧑🏿‍💻", "✨", "🫠", "⚽", "🍣", "1️⃣", "©️"]


def prose(rng, fakers):
    fake = rng.choice(fakers)
    kind = rng.randrange(6)
    if kind == 0:
        return fake.name()
    if kind == 1:
        return fake.address()
    if kind == 2:
        return fake.text(max_nb_chars=rng.choice([60, 200, 600, 1500]))
    if kind == 3:
        return f"{fake.company()}: {fake.sentence()}"
    if kind == 4:
        return " ".join(fake.words(rng.randrange(1, 12)))
    return fake.paragraph(nb_sentences=rng.randrange(1, 6))


def code(rng, sources):
    lines = rng.choice(sources)
    start = rng.randrange(max(1, len(lines) - 20))
    return "\n".join(lines[start : start + rng.randrange(1, 20)])


def oddities(rng):
    pieces = [
        " " * rng.randrange(1, 9),
        "\t" * rng.randrange(1, 4),
        "\n" * rng.randrange(1, 4),
        "\r\n",
        "".join(rng.choice(EMOJI) for _ in range(rng.randrange(1, 6))),
        "".join(chr(rng.choice([rng.randrange(0x20, 0xD7FF), rng.randrange(0xE000, 0xFFFD), rng.randrange(0x10000, 0x2FFFF)])) for _ in range(rng.randrange(1, 8))),
        "é ä ñ",
        "3.14159 -42 1e9 0xFF 1,000,000",
        "https://example.com/path?q=1&r=two#frag",
        "▁ <0x41> <bos> <start_of_turn>",
    ]
    return "".join(rng.choice(pieces) for _ in range(rng.randrange(1, 5)))


def strings():
    from faker import Faker

    rng = random.Random(4)
    fakers = []
    for locale in LOCALES:
        fake = Faker(locale)
        fake.seed_instance(rng.randrange(1 << 30))
        fakers.append(fake)
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    sources = [p.read_text(errors="replace").splitlines() for p in sorted(stdlib.glob("*.py"))[:80]]
    out = []
    for _ in range(COUNT):
        kind = rng.random()
        s = prose(rng, fakers) if kind < 0.7 else code(rng, sources) if kind < 0.85 else oddities(rng)
        if rng.random() < 0.2:
            s = oddities(rng) + s + oddities(rng)
        out.append(s)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    STRINGS.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in out), encoding="utf-8")
    print(f"{len(out)} strings, {STRINGS.stat().st_size} bytes -> {STRINGS}")


def load():
    return [json.loads(line) for line in STRINGS.read_text(encoding="utf-8").splitlines()]


def toy():
    """A small tokenizer shaped like Gemma's: spaces become ▁, no pre-tokenizer, BPE with byte fallback."""
    from tokenizers import Tokenizer, decoders, models, normalizers, trainers

    tok = Tokenizer(models.BPE(byte_fallback=True, fuse_unk=True, unk_token="<unk>"))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    special = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    alphabet = [f"<0x{b:02X}>" for b in range(256)]
    trainer = trainers.BpeTrainer(vocab_size=4000, special_tokens=special + alphabet, limit_alphabet=600, max_token_length=16)
    tok.train_from_iterator(load()[: COUNT // 2], trainer)
    doc = json.loads(tok.to_str())
    doc["added_tokens"] = [t for t in doc["added_tokens"] if t["content"] in special]  # byte tokens: vocabulary
    (FIXTURES / "toy.json").write_text(json.dumps(doc, ensure_ascii=False))
    print(f"{tok.get_vocab_size()} tokens -> {FIXTURES / 'toy.json'}")


def encode(path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    tok.encode_special_tokens = True
    return tok, [tok.encode(s, add_special_tokens=False).ids for s in load()]


def fnv(ids):
    h = 0xCBF29CE484222325
    for i in ids:
        for b in i.to_bytes(4, "little"):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def ids(path, name):
    _, encoded = encode(path)
    (FIXTURES / f"{name}.ids").write_text("".join(f"{len(e)} {fnv(e):016x}\n" for e in encoded))
    print(f"{sum(map(len, encoded))} tokens over {len(encoded)} strings -> {FIXTURES / (name + '.ids')}")


def show(path, index):
    tok, encoded = encode(path)
    e = encoded[int(index)]
    print(json.dumps(load()[int(index)], ensure_ascii=False))
    print(e)
    print([tok.id_to_token(i) for i in e])


CONVERSATIONS = [
    ([("user", "Hello!")], False),
    ([("system", "You are terse."), ("user", "Why is the sky blue?")], False),
    ([("user", "  Think about 17 × 23.  ")], True),
    ([("system", " Answer in Spanish.\n"), ("user", "¿Qué hora es?"), ("assistant", "Son las tres. "), ("user", "¿Y en Tokio?")], True),
    ([("user", "Tell me a joke"), ("assistant", "Why did the "), ("assistant", "GPU blush?"), ("user", "why?")], False),
    ([("user", "日本語で答えて 🤖\n\n改行も"), ("system", "Be kind."), ("user", "ok\tthen")], False),
]


def chat(path, template, name):
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    render = env.from_string(Path(template).read_text())
    tok = Tokenizer.from_file(path)
    lines = []
    for messages, think in CONVERSATIONS:
        msgs = [{"role": r, "content": c} for r, c in messages]
        text = render.render(messages=msgs, bos_token="<bos>", add_generation_prompt=True, enable_thinking=think)
        ids = tok.encode(text, add_special_tokens=False).ids
        lines.append(json.dumps({"messages": msgs, "think": think, "text": text, "ids": ids}, ensure_ascii=False))
    (FIXTURES / f"{name}.chat").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} conversations -> {FIXTURES / (name + '.chat')}")


if __name__ == "__main__":
    {"strings": strings, "toy": toy, "ids": ids, "show": show, "chat": chat}[sys.argv[1]](*sys.argv[2:])