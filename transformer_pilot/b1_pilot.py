"""
B1 pilot: self vs. other in a transformer, with outsiders.

Roles
  target    gpt2-medium (355M)  - the "self": writes continuations
  inserter  gpt2-large (774M)   - writes the "other" tokens an experimenter inserts
  outsiders copy      = a second, independently loaded gpt2-medium (full access)
            naive     = gpt2 small (124M), never sees the target's outputs
            distilled = distilgpt2 (82M) fine-tuned only on the target's outputs

Part 1  Exact-zero: are the target's activations after a token it wrote itself
        identical to those after the same token fed in by an experimenter?
Part 2  Self/other detection: at 4 positions per continuation, the token is either the
        target's own or an inserted one (fluent = inserter's choice given the same prefix;
        off-context = inserter's choice for a different prompt). Score = log P(token | prefix).
        AUROC: 1 = perfect separation of own from inserted, 0.5 = chance.
Part 3  Outsiders: the same scores computed by each outsider. AUROC plus Spearman
        correlation with the target's own score (how well an outsider reproduces the
        target's judgment). The distilled outsider is measured after 0, 1 and 3 epochs.
"""
import json, re, time, random
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "transformer_pilot")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(0); random.seed(0); np.random.seed(0)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

M_TARGET, M_INS = "openai-community/gpt2-medium", "openai-community/gpt2-large"
M_NAIVE, M_DIST = "openai-community/gpt2", "distilbert/distilgpt2"
N_TEST, N_TRAIN, CONT, POS = 60, 400, 32, [6, 14, 22, 30]
tok = AutoTokenizer.from_pretrained(M_TARGET)            # all four share GPT-2's tokenizer
tok.pad_token = tok.eos_token


def clean(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([.,;:!?)\]'])", r"\1", s)
    s = re.sub(r"([(\[]) ", r"\1", s)
    return s.replace(" n't", "n't").replace(" 's", "'s")


df = pd.read_parquet(hf_hub_download("Salesforce/wikitext",
                     "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))
lines = [clean(l.strip()) for l in df.text if len(l.split()) > 40 and not l.strip().startswith("=")]
random.shuffle(lines)
prompts = [" ".join(l.split()[:16]) for l in lines]
test_p, train_p = prompts[:N_TEST], prompts[N_TEST:N_TEST + N_TRAIN]
assert not set(test_p) & set(train_p)

load = lambda m: AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32).eval()
target = load(M_TARGET)


@torch.no_grad()
def logp(model, prefix_ids, token_id):
    out = model(torch.tensor([prefix_ids]))
    return torch.log_softmax(out.logits[0, -1], -1)[token_id].item()


@torch.no_grad()
def sample_next(model, prefix_ids, g):
    logits = model(torch.tensor([prefix_ids])).logits[0, -1]
    v, i = torch.topk(logits, 50)
    return i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)].item()


# ---------------- Part 1: exact-zero ----------------
log("Part 1: exact-zero check")
g = torch.Generator().manual_seed(1)
p1 = []
with torch.no_grad():
    for prompt in test_p[:10]:
        ids = tok(prompt)["input_ids"]
        # (a) the target writes 16 tokens itself, recording its last-layer state at each step
        out = target(torch.tensor([ids]), use_cache=True, output_hidden_states=True)
        past, h_self, own = out.past_key_values, [], []
        nxt = sample_next(target, ids, g)
        for _ in range(16):
            own.append(nxt)
            o = target(torch.tensor([[nxt]]), past_key_values=past, use_cache=True,
                       output_hidden_states=True)
            past = o.past_key_values; h_self.append(o.hidden_states[-1][0, -1].clone())
            v, i = torch.topk(o.logits[0, -1], 50)
            nxt = i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)].item()
        # (b) an experimenter feeds the identical tokens in, one at a time
        out = target(torch.tensor([ids]), use_cache=True)
        past, h_fed = out.past_key_values, []
        for t in own:
            o = target(torch.tensor([[t]]), past_key_values=past, use_cache=True,
                       output_hidden_states=True)
            past = o.past_key_values; h_fed.append(o.hidden_states[-1][0, -1].clone())
        # (c) the experimenter pastes the whole text in one pass (a different computation path)
        full = target(torch.tensor([ids + own]), output_hidden_states=True).hidden_states[-1][0]
        h_paste = full[len(ids):]
        A, B = torch.stack(h_self), torch.stack(h_fed)
        p1.append(dict(fed=(A - B).abs().max().item(),
                       paste=(A - h_paste).abs().max().item(),
                       scale=A.abs().mean().item()))
part1 = {k: max(d[k] for d in p1) for k in ("fed", "paste")}
part1["typical_activation"] = float(np.mean([d["scale"] for d in p1]))
log("  max |self - fed| =", part1["fed"], "| max |self - pasted| =", part1["paste"],
    "| typical |activation| =", round(part1["typical_activation"], 3))

# ---------------- Part 2: build the self/other set ----------------
log("Part 2: target continuations and insertions")
inserter = load(M_INS)
g = torch.Generator().manual_seed(2)
items, conts = [], []
for prompt in test_p:
    ids = tok(prompt)["input_ids"]
    cont = []
    for _ in range(CONT):
        cont.append(sample_next(target, ids + cont, g))
    conts.append((ids, cont))
fluent = {}
for k, (ids, cont) in enumerate(conts):
    for p in POS:
        prefix = ids + cont[:p]
        fluent[(k, p)] = sample_next(inserter, prefix, g)
for k, (ids, cont) in enumerate(conts):
    for p in POS:
        prefix, own = ids + cont[:p], cont[p]
        items.append(dict(k=k, p=p, prefix=prefix, tok=own, label=1, kind="self"))
        if fluent[(k, p)] != own:
            items.append(dict(k=k, p=p, prefix=prefix, tok=fluent[(k, p)], label=0, kind="fluent"))
        off = fluent[((k + 1) % N_TEST, p)]               # chosen for a different prompt
        if off != own:
            items.append(dict(k=k, p=p, prefix=prefix, tok=off, label=0, kind="offctx"))
del inserter
log("  items:", pd.Series([i["kind"] for i in items]).value_counts().to_dict())


def score_all(model):
    return np.array([logp(model, it["prefix"], it["tok"]) for it in items])


kinds = np.array([i["kind"] for i in items]); labels = np.array([i["label"] for i in items])
groups = np.array([i["k"] for i in items])


def auroc_ci(s, kind, B=1000):
    m = (kinds == "self") | (kinds == kind)
    est = roc_auc_score(labels[m], s[m])
    rng, ks, vals = np.random.default_rng(0), np.unique(groups), []
    for _ in range(B):                                    # bootstrap over prompts
        pick = rng.choice(ks, len(ks))
        idx = np.concatenate([np.where(m & (groups == q))[0] for q in pick])
        if len(set(labels[idx])) == 2:
            vals.append(roc_auc_score(labels[idx], s[idx]))
    return est, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


scores = {"target (self)": score_all(target)}
log("Part 3: outsiders")
copy = load(M_TARGET); scores["copy (full access)"] = score_all(copy); del copy
naive = load(M_NAIVE); scores["naive (gpt2 small)"] = score_all(naive); del naive

# ---------------- distilled outsider: trained only on the target's outputs ----------------
log("  generating target outputs on held-out prompts for distillation")
train_txt = []
tok.padding_side = "left"
with torch.no_grad():
    for b in range(0, N_TRAIN, 16):
        enc = tok(train_p[b:b + 16], return_tensors="pt", padding=True)
        gen = target.generate(**enc, max_new_tokens=48, do_sample=True, top_k=50,
                              pad_token_id=tok.eos_token_id)
        for row, m in zip(gen, enc["attention_mask"]):
            plen = int(m.sum()); full = row[len(m) - plen:].tolist()
            train_txt.append((full[:plen], full[plen:]))
log("  distillation set:", len(train_txt), "continuations")

dist = load(M_DIST)
scores["distilled, 0 epochs"] = score_all(dist)
opt = torch.optim.AdamW(dist.parameters(), lr=5e-5)
for ep in range(3):
    dist.train(); random.shuffle(train_txt)
    for b in range(0, len(train_txt), 8):
        batch = train_txt[b:b + 8]; L = max(len(p) + len(c) for p, c in batch)
        x = torch.full((len(batch), L), tok.eos_token_id); y = torch.full((len(batch), L), -100)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for r, (p, c) in enumerate(batch):
            s = p + c; x[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
            y[r, len(p):len(s)] = torch.tensor(c)             # loss on the target's words only
        loss = dist(input_ids=x, attention_mask=att, labels=y).loss
        opt.zero_grad(); loss.backward(); opt.step()
    dist.eval()
    log(f"  epoch {ep+1} last loss {loss.item():.3f}")
    if ep + 1 in (1, 3):
        scores[f"distilled, {ep+1} epoch{'s' if ep else ''}"] = score_all(dist)

# ---------------- report ----------------
ref = scores["target (self)"]
rows = []
for name, s in scores.items():
    f = auroc_ci(s, "fluent"); o = auroc_ci(s, "offctx")
    rows.append(dict(scorer=name,
                     auroc_fluent=f"{f[0]:.3f} [{f[1]:.3f}-{f[2]:.3f}]",
                     auroc_offctx=f"{o[0]:.3f} [{o[1]:.3f}-{o[2]:.3f}]",
                     spearman_with_self=f"{spearmanr(s, ref).correlation:.3f}",
                     max_abs_diff_from_self=f"{np.abs(s - ref).max():.2e}"))
rep = pd.DataFrame(rows)
print("\n" + rep.to_string(index=False))
np.savez(os.path.join(OUT, "b1_scores.npz"), labels=labels, kinds=kinds, groups=groups,
         **{re.sub(r"\W+", "_", k): v for k, v in scores.items()})   # for paired tests later
json.dump(dict(part1=part1, table=rows,
               counts=pd.Series(kinds).value_counts().to_dict()), open(os.path.join(OUT, "b1_results.json"), "w"), indent=1)
log("done")
