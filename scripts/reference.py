#!/usr/bin/env python3
"""Writes the reference of the engine's top-1 agreement test (tests/engine.rs): a Gemma 4 model's greedy answers in
bfloat16, as Hugging Face computes them. One JSON line per prompt: the prompt's text, its tokens through the chat
template, the answer's tokens, and at each the gap between the best logit and the runner-up. The fixture is named
after the model's directory in Tachyon's home (`gemma-4-e4b`, `gemma-4-12b`, `gemma-4-31b`, ...).

Needs torch, transformers and accelerate. The model spreads over the GPUs it sees and, past their memory, the CPU
(the 31B's 62 GB of bfloat16 on a 48 GB GPU: slow, but a reference is made once):

    python3 scripts/reference.py ~/models/gemma-4-E4B-it > crates/tachyon/tests/fixtures/gemma-4-e4b.ref
"""

import json
import sys
import time
import warnings

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ANSWER = 64
PROMPTS = [
    "Why is the sky blue?",
    "Write a haiku about autumn rain.",
    "What is the capital of Australia?",
    "Explain recursion to a ten-year-old.",
    "Give me three tips for a job interview.",
    "Translate 'Where is the train station?' into Spanish.",
    "What is 17 times 23?",
    "Write a Python function that reverses a string.",
    "Who wrote Pride and Prejudice?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What causes the seasons on Earth?",
    "List the planets of the solar system in order.",
    "How do vaccines work?",
    "Suggest a name for a black cat.",
    "What is the difference between weather and climate?",
    "Write a short apology email for missing a meeting.",
    "How many legs does a spider have?",
    "Explain what a database index is.",
    "What is photosynthesis?",
    "Tell me a joke about programmers.",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "What are the main causes of the French Revolution?",
    "Describe the taste of a lemon.",
    "How do I boil an egg?",
    "What does HTTP stand for?",
    "Write a limerick about a cat who loves cheese.",
    "Why do we dream?",
    "What is the Pythagorean theorem?",
    "Give an example of a metaphor.",
    "How does a bicycle stay upright?",
    "¿Cuál es el río más largo del mundo?",
    "Explique la différence entre un virus et une bactérie.",
]


def main(src):
    # PyTorch warns when it cannot reach the driver's management library (NVML); CUDA itself works.
    warnings.filterwarnings("ignore", message="Can't initialize NVML")
    start = time.time()
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="auto").eval()
    for text in PROMPTS:
        chat = [{"role": "user", "content": text}]
        prompt = list(tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
        ids = torch.tensor([prompt], device="cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=ANSWER, do_sample=False, output_logits=True, return_dict_in_generate=True)
        answer = out.sequences[0, len(prompt):].tolist()
        top = [torch.topk(l[0].float(), 2).values.tolist() for l in out.logits]
        print(json.dumps({"text": text, "prompt": prompt, "answer": answer, "gap": [round(a - b, 3) for a, b in top]}, ensure_ascii=False))
        done = PROMPTS.index(text) + 1
        print(f"{done}/{len(PROMPTS)} prompts, {time.time() - start:.0f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])