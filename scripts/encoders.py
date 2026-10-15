#!/usr/bin/env python3
"""Step 10's references and precision study for the two encoders: Laya multilingual (decisions) and Granite
Embedding 311M Multilingual R2 (embeddings), both ModernBERT on mmBERT-base's shape.

Writes, in float32 as the `laya` package and sentence-transformers compute them (TF32 off):

    crates/tachyon/tests/fixtures/laya-multilingual.ref  one JSON line per (state, question): the inputs, the
                                                         sequence Laya builds, its option logits and probabilities
    crates/tachyon/tests/fixtures/granite-311m.ref       one JSON line per text: its tokens and its embedding

then runs both again with the encoder's matrix products in bf16 (what `laya` runs on a GPU), in f16, in Q8 (8-bit weights and activations, a scale per
64 values, the GEMM Tachyon has) and in Q4 weights with Q8 activations, and prints how far each moves the answers.

Needs a CUDA GPU, the `laya` package (which brings torch and transformers 5) and network access to the Hub:

    pip install laya sentence-transformers
    python3 scripts/encoders.py

`python3 scripts/encoders.py laya REPO NAME` and `python3 scripts/encoders.py granite REPO NAME` do the same for one
other checkpoint (a Hub id or a directory), into NAME.ref:

    python3 scripts/encoders.py laya convaiinnovations/laya laya-english
    python3 scripts/encoders.py laya convaiinnovations/laya-typed-decisions laya-typed-decisions
    python3 scripts/encoders.py granite ibm-granite/granite-embedding-97m-multilingual-r2 granite-97m

`python3 scripts/encoders.py tiny` instead writes crates/tachyon/tests/fixtures/tiny-encoder/: a small random Laya
(the package's own model on a 3-layer ModernBERT, weights in f16) and its outputs in float32, for the tests that
run without the real checkpoints. It needs only the CPU.
"""

import json
import os
import random
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "tachyon", "tests", "fixtures")
CASES = 500

# States: short messages in twelve languages, JSON records, a conversation (truncated from the left), and long
# texts past the 128-token sliding window and past Laya's 1024-token budget.
STATES = [
    "Hi, I was charged twice for my subscription this month. Please refund the second payment.",
    "The app crashes every time I open the camera. I'm on Android 14, Pixel 8.",
    "Can you tell me when my order #48213 will arrive? It has been ten days.",
    "I love the new update, the dark mode looks great! Thanks team.",
    "I want to close my account and delete all my data. This is my third request.",
    "Hola, no puedo iniciar sesión desde ayer. Me dice que mi contraseña es incorrecta pero la cambié.",
    "Quiero cancelar mi suscripción, el servicio es muy lento y nadie responde a mis correos.",
    "Guten Tag, die Lieferung kam beschädigt an. Ich möchte eine Rückerstattung oder einen Ersatz.",
    "Ich bin sehr zufrieden mit dem Kundenservice, vielen Dank für die schnelle Hilfe!",
    "Bonjour, ma facture contient une erreur : on m'a facturé 49 € au lieu de 29 €.",
    "Je n'arrive pas à connecter l'imprimante au Wi-Fi, j'ai suivi le guide sans succès.",
    "Olá, gostaria de saber se vocês entregam em Portugal e quanto custa o envio.",
    "Estou muito insatisfeito, o produto parou de funcionar depois de uma semana.",
    "Ciao, vorrei cambiare l'indirizzo di consegna del mio ordine prima della spedizione.",
    "Goedemorgen, mijn wachtwoord werkt niet meer en ik krijg geen resetmail.",
    "您好，我上个月的账单多扣了一笔费用，请帮我退款。",
    "应用程序一直闪退，我已经重新安装了两次，还是不行。",
    "注文した商品がまだ届きません。追跡番号を教えていただけますか？",
    "サービスにとても満足しています。いつもありがとうございます。",
    "مرحبا، لا أستطيع الدخول إلى حسابي منذ أمس، أرجو المساعدة بسرعة.",
    "أريد إلغاء اشتراكي واسترداد المبلغ المدفوع لهذا الشهر.",
    "Здравствуйте, мне дважды списали деньги за заказ. Верните, пожалуйста, лишний платёж.",
    "Отличный сервис, всё пришло вовремя и в хорошем состоянии!",
    "नमस्ते, मेरा ऑर्डर अभी तक नहीं आया है, कृपया जल्दी बताएं कि यह कब आएगा।",
    "मुझे अपना खाता बंद करना है और मेरा सारा डेटा हटाना है।",
    "URGENT: our production database is down and customers cannot log in. Need help immediately!!!",
    "Just wanted to say the onboarding call yesterday was really helpful.",
    "Can I get an invoice with my company's VAT number for last year's purchases?",
    "Your website says free shipping over $50 but I was charged $7.99 on a $62 order.",
    "My email is jane.doe@example.com and my phone is +1 415 555 0134, please call me back.",
    "안녕하세요, 주문한 상품이 아직 도착하지 않았습니다. 언제 받을 수 있나요?",
    "Merhaba, kartımdan iki kez ödeme alındı, lütfen fazla tutarı iade edin.",
    "Dzień dobry, aplikacja nie działa po ostatniej aktualizacji, ciągle się zawiesza.",
    "Xin chào, tôi muốn hủy đơn hàng và nhận lại tiền.",
    "Halo, saya tidak bisa masuk ke akun saya sejak kemarin, mohon bantuannya.",
    {"channel": "email", "from": "maria@example.es", "subject": "Factura duplicada",
     "body": "Me han cobrado dos veces la misma factura de septiembre. Necesito el reembolso."},
    {"ticket": 7731, "priority": "unset", "customer": {"plan": "enterprise", "seats": 250},
     "text": "SSO login fails for all users since the certificate rotation this morning."},
    {"order": {"id": "A-1029", "items": 3, "total": 129.5, "currency": "EUR"},
     "message": "Two of the three items are missing from the package."},
    [{"role": "customer", "text": "Hi, my card was declined but the money left my account."},
     {"role": "agent", "text": "Sorry to hear that. Can you share the last four digits of the card?"},
     {"role": "customer", "text": "It ends in 4421. The charge was 89.90."},
     {"role": "agent", "text": "Thanks, I see a pending authorization that should be released in 3 to 5 days."},
     {"role": "customer", "text": "That's not acceptable, I need that money today. I want to cancel everything."}],
]
PARAGRAPH = (
    "Our team has reviewed the incident report submitted last week regarding the intermittent outages in the "
    "European region. The root cause appears to be a misconfigured load balancer that drops connections when "
    "traffic spikes above a threshold during the evening peak. We have applied a temporary fix, but customers "
    "may still notice slow responses until the permanent change is deployed next Tuesday. "
)
LONG = [
    PARAGRAPH * 4 + "Please confirm whether this explanation is enough for your compliance review.",
    PARAGRAPH * 14 + "Finally, I would like to cancel the contract and get a refund for the downtime.",
    ("Estimado equipo, " + "les escribo para informarles de que el pedido llegó incompleto y la caja estaba dañada. "
     * 12 + "Solicito el reembolso completo."),
]

QUESTIONS = [
    {"type": "choice", "instructions": "What is the customer's main issue?",
     "criteria": {"billing": "charges, invoices, refunds", "technical": "bugs, crashes, connectivity",
                  "account": "login, password, data deletion", "shipping": "delivery and orders",
                  "other": "anything else"}},
    {"type": "choice", "instructions": "What is the sentiment of the message?",
     "criteria": ["positive", "neutral", "negative"]},
    {"type": "score", "instructions": "How urgent is this request?",
     "criteria": ["low", "medium", "high", "critical"]},
    {"type": "noul", "instructions": "The customer asks for a refund."},
    {"type": "noul", "instructions": "The message contains personal data.",
     "criteria": {"true": "an email address, phone number, card or account number",
                  "false": "no personal identifiers"}},
    {"type": "choice", "instructions": "Which department should handle this?",
     "criteria": ["sales", "support", "legal", "hr", "finance", "logistics"]},
    {"type": "score", "instructions": "How satisfied is the customer?",
     "criteria": ["very unsatisfied", "unsatisfied", "neutral", "satisfied", "very satisfied"]},
    {"type": "noul", "instructions": "The writer is angry."},
    {"type": "choice", "instructions": "Is this message spam?",
     "criteria": {"spam": "unsolicited advertising or scams", "not_spam": "a genuine customer message"}},
    {"type": "choice", "instructions": "Which topic fits best?",
     "criteria": ["payments", "refunds", "invoices", "login", "passwords", "privacy", "crashes", "performance",
                  "delivery", "returns", "damaged goods", "praise"]},
    {"type": "choice", "instructions": "Pick the product area.",
     "criteria": {"area_%02d" % i: "a detailed description of product area number %d with its typical problems" % i
                  for i in range(30)}},
    {"type": "noul", "instructions": "¿El cliente quiere cancelar su suscripción o contrato?"},
]

QUERIES = [
    "how do I get a refund", "app keeps crashing", "¿dónde está mi pedido?", "Passwort zurücksetzen",
    "facture incorrecte", "如何取消订阅", "配送状況を確認したい", "استرداد المبلغ", "двойное списание",
    "खाता बंद करना", "Rust borrow checker lifetime error", "SELECT * FROM orders WHERE total > 100",
]


def text_of(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def laya_cases():
    states = STATES + LONG
    pairs = [(s, q) for s in states for q in QUESTIONS]
    random.Random(10).shuffle(pairs)
    return pairs[:CASES]


# ------------------------------------------------------------------------------------------------ quantization
def q8(x, group=64):
    """Symmetric 8-bit with a scale per `group` consecutive values along the last axis."""
    shape = x.shape
    g = x.reshape(-1, group)
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    return (torch.round(g / scale).clamp(-127, 127) * scale).reshape(shape)


def q4(x, group=64):
    """4-bit with zero point 8 (levels -8..7) and a scale per `group` values, as Tachyon's Q4."""
    shape = x.shape
    g = x.reshape(-1, group)
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7.5
    return (torch.round(g / scale).clamp(-8, 7) * scale).reshape(shape)


def f16(x):
    return x.half().float()


def bf16(x):
    return x.bfloat16().float()


# bf16 is the precision `laya` runs in on a GPU by default (`amp_dtype` in its config), for comparison.
FORMATS = {"bf16": (bf16, bf16), "f16": (f16, f16), "q8": (q8, q8), "q4": (q4, q8)}


class Quantized:
    """Within the block, every linear layer of the encoder's transformer layers sees its weight and its input
    rounded to `fmt`; everything else stays float32."""

    def __init__(self, encoder, fmt):
        self.linears = [m for name, m in encoder.named_modules() if isinstance(m, torch.nn.Linear) and "layers." in name]
        self.wq, self.aq = FORMATS[fmt]
        self.saved, self.hooks = [], []

    def __enter__(self):
        for m in self.linears:
            self.saved.append(m.weight.data)
            m.weight.data = self.wq(m.weight.data)
            self.hooks.append(m.register_forward_pre_hook(lambda _, args: (self.aq(args[0]),) + args[1:]))
        return self

    def __exit__(self, *_):
        for m, w in zip(self.linears, self.saved):
            m.weight.data = w
        for h in self.hooks:
            h.remove()


# ------------------------------------------------------------------------------------------------ Laya
def laya_items(agent, pairs):
    items = []
    for state, qdef in pairs:
        agent._check_question("q", qdef)
        internal = {"q": agent._to_internal(qdef)}
        items.append(agent._encode_state(state, ["q"], internal)[0])
    return items


@torch.no_grad()
def laya_logits(agent, items):
    out = []
    dev = agent.device
    for it in items:
        ids = torch.tensor([it["ids"]], device=dev)
        k = len(it["markers"])
        logits, _ = agent.model(ids, torch.ones_like(ids), torch.tensor([it["markers"]], device=dev),
                                torch.ones((1, k), dtype=torch.bool, device=dev), torch.tensor([it["qtype"]], device=dev))
        out.append(logits[0, :k].double().cpu())
    return out


def laya_probs(agent, logits, qtype):
    from laya.common import temp_bucket
    k = len(logits)
    t = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    return torch.softmax(logits / t, -1)


# ------------------------------------------------------------------------------------------------ Granite
@torch.no_grad()
def granite_embeddings(model, batches):
    out = []
    for enc in batches:
        h = model(**enc).last_hidden_state[:, 0]
        out.append(F.normalize(h.double(), dim=-1)[0].cpu())
    return out


def laya_reference(repo, name):
    """Writes `name`.ref for the Laya checkpoint `repo` (a Hub id or a directory); returns the agent, its items and
    their float32 logits."""
    import laya
    agent = laya.load(repo, device="cuda")
    agent.model.float().eval()
    cfg = agent.model.encoder.config
    print("%s: rope %s, window %s, layers %s, temperatures %s %s" % (
        name, getattr(cfg, "rope_parameters", None), cfg.local_attention, cfg.layer_types[:3], agent.temperature,
        agent.temperature_by_options))
    pairs = laya_cases()
    items = laya_items(agent, pairs)
    ref = laya_logits(agent, items)
    with open(os.path.join(ROOT, name + ".ref"), "w") as f:
        for (state, q), it, lg in zip(pairs, items, ref):
            p = laya_probs(agent, lg, it["qtype"])
            f.write(json.dumps({"state": state, "question": q, "ids": it["ids"], "markers": it["markers"],
                                "qtype": it["qtype"], "logits": [round(float(v), 6) for v in lg],
                                "probs": [round(float(v), 7) for v in p]}, ensure_ascii=False) + "\n")
    print("%s: %d cases, %d to %d tokens" % (name, len(items), min(len(i["ids"]) for i in items),
                                             max(len(i["ids"]) for i in items)))
    return agent, items, ref


def granite_reference(repo, name):
    """Writes `name`.ref for the Granite Embedding checkpoint `repo`; returns the model, its inputs and their float32
    embeddings."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo, torch_dtype=torch.float32).cuda().eval()
    print("%s: %s" % (name, {k: getattr(model.config, k, None) for k in ("global_rope_theta", "local_rope_theta",
                                                                          "rope_parameters", "local_attention", "norm_eps", "hidden_activation")}))
    texts = [text_of(s) for s in STATES + LONG] + QUERIES
    batches = [tok(t, return_tensors="pt").to("cuda") for t in texts]
    gref = granite_embeddings(model, batches)
    with open(os.path.join(ROOT, name + ".ref"), "w") as f:
        for t, enc, e in zip(texts, batches, gref):
            f.write(json.dumps({"text": t, "ids": enc["input_ids"][0].tolist(),
                                "embedding": [round(float(v), 7) for v in e]}, ensure_ascii=False) + "\n")
    print("%s: %d texts, %d to %d tokens" % (name, len(texts), min(b["input_ids"].shape[1] for b in batches),
                                             max(b["input_ids"].shape[1] for b in batches)))
    return model, batches, gref


def study(laya_run=None, granite_run=None):
    """How far each format of `FORMATS` moves Laya's answers and Granite's embeddings from float32."""
    print("\n%-6s %12s %12s %10s %14s %14s" % ("format", "max |dp|", "mean |dp|", "top-1", "min cosine", "mean cosine"))
    for fmt in FORMATS:
        row = [fmt]
        if laya_run:
            agent, items, ref = laya_run
            with Quantized(agent.model.encoder, fmt):
                got = laya_logits(agent, items)
            dp = [(laya_probs(agent, g, it["qtype"]) - laya_probs(agent, r, it["qtype"])).abs()
                  for g, r, it in zip(got, ref, items)]
            top = sum(int(g.argmax() == r.argmax()) for g, r in zip(got, ref)) / len(ref)
            row += ["%12.2e" % max(float(d.max()) for d in dp), "%12.2e" % float(torch.cat(dp).mean()), "%9.1f%%" % (100 * top)]
        else:
            row += ["%12s" % "", "%12s" % "", "%10s" % ""]
        if granite_run:
            model, batches, gref = granite_run
            with Quantized(model, fmt):
                emb = granite_embeddings(model, batches)
            cos = torch.tensor([float((a * b).sum()) for a, b in zip(emb, gref)])
            row += ["%14.6f" % float(cos.min()), "%14.6f" % float(cos.mean())]
        print("%-6s " % row[0] + " ".join(row[1:]))


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import laya
    import transformers
    print("laya %s, transformers %s, torch %s" % (getattr(laya, "__version__", "?"), transformers.__version__, torch.__version__))
    if not args:
        study(laya_reference("convaiinnovations/laya-multilingual", "laya-multilingual"),
              granite_reference("ibm-granite/granite-embedding-311m-multilingual-r2", "granite-311m"))
    elif args[0] == "laya" and len(args) == 3:
        study(laya_run=laya_reference(args[1], args[2]))
    elif args[0] == "granite" and len(args) == 3:
        study(granite_run=granite_reference(args[1], args[2]))
    else:
        sys.exit(__doc__)


@torch.no_grad()
def tiny():
    """A random Laya on a ModernBERT small enough to store: a global layer and two sliding ones over a 16-token
    window with their own RoPE bases, 4 heads, a 2-head decision head; three sequences, the longest past the window."""
    from laya.common import DecisionModel
    from safetensors.torch import save_file
    from transformers import ModernBertConfig, ModernBertModel
    torch.manual_seed(10)
    out = os.path.join(ROOT, "tiny-encoder")
    os.makedirs(out, exist_ok=True)
    cfg = ModernBertConfig(vocab_size=64, hidden_size=128, intermediate_size=128, num_hidden_layers=3,
                           num_attention_heads=4, local_attention=16, global_attn_every_n_layers=3,
                           layer_types=["full_attention", "sliding_attention", "sliding_attention"],
                           rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                                            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
                           hidden_activation="gelu", norm_eps=1e-5, pad_token_id=0, bos_token_id=2, eos_token_id=1,
                           cls_token_id=2, sep_token_id=1, max_position_embeddings=512)
    model = DecisionModel(ModernBertModel(cfg), head_layers=2, n_act=2).eval()
    for p in model.parameters():  # trained-looking scales, stored exactly in f16
        p.copy_((p + 0.02 * torch.randn_like(p)).half().float())
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(cfg.to_json_string())
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump({"head_layers": 2, "max_len": 64, "head_max_len": 24, "temperature": [1.0, 1.0, 1.0],
                   "temperature_by_options": {}}, f)
    save_file({k: v.half().contiguous() for k, v in model.state_dict().items()}, os.path.join(out, "model.safetensors"))
    rng = random.Random(10)
    with open(os.path.join(out, "outputs.ref"), "w") as f:
        for n, k, qtype in [(7, 2, 2), (24, 3, 0), (40, 5, 1)]:
            ids = [2] + [rng.randrange(5, 64) for _ in range(n - 2)] + [1]
            markers = sorted(rng.sample(range(1, n - 1), k))
            for m in markers:
                ids[m] = 4
            t = torch.tensor([ids])
            hidden = model.encoder(input_ids=t, attention_mask=torch.ones_like(t)).last_hidden_state[0]
            logits, _ = model(t, torch.ones_like(t), torch.tensor([markers]), torch.ones((1, k), dtype=torch.bool),
                              torch.tensor([qtype]))
            f.write(json.dumps({"ids": ids, "markers": markers, "qtype": qtype,
                                "hidden": [round(float(v), 6) for v in hidden.flatten()],
                                "logits": [round(float(v), 6) for v in logits[0]]}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(tiny() if sys.argv[1:] == ["tiny"] else main(sys.argv[1:]))
