"""
B2: self vs. other in a transformer, with outsiders trained on DIFFERENT data. GPU version.

Fixes B1's two gaps:
  1. B1's outsiders all shared GPT-2's training data (WebText). B2 adds GPT-Neo 125M and 1.3B,
     which use the same tokenizer but were trained on the Pile.
  2. B1 had no paired test. B2 saves per-token scores and reports paired bootstrap intervals
     for AUROC(self) - AUROC(outsider), resampling prompts.
Also: 150 test prompts (B1: 60) and 4,000 distillation continuations (B1: 400).

Roles
  target    gpt2-medium (355M)       the "self"
  inserter  gpt2-large (774M)        writes the inserted "other" tokens
  outsiders copy           second load of gpt2-medium (full access)
            gpt2-small     shared training data, no access
            neo-125m       different training data, no access
            neo-1.3b       different training data, no access, 4x the target's size
            neo-125m-dist  different data, then fine-tuned on the target's outputs only

Score = log P(token | prefix). AUROC: 1 = perfect separation of own from inserted, 0.5 = chance.
Writes results to results/transformer_pilot/. Model files (about 11 GB) are cached in .hf_cache at the repository root,
which is git-ignored and can be deleted after a run.
"""
import os, json, re, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

OUT = os.path.join(ROOT, "results", "transformer_pilot", "smoke" if os.environ.get("B2_SMOKE") == "1" else "")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0); random.seed(0); np.random.seed(0)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
log("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")

M = dict(target="openai-community/gpt2-medium", inserter="openai-community/gpt2-large",
         small="openai-community/gpt2", neo125="EleutherAI/gpt-neo-125m",
         neo13="EleutherAI/gpt-neo-1.3B")
SMOKE = os.environ.get("B2_SMOKE") == "1"          # tiny run to test the pipeline
N_TEST, N_TRAIN, CONT, POS = (6, 32, 32, [6, 14, 22, 30]) if SMOKE else (150, 4000, 32, [6, 14, 22, 30])
from huggingface_hub import snapshot_download
SMOKE_SKIP = {"neo13"} if os.environ.get("B2_SMOKE") == "1" else set()
for key, repo in M.items():                       # fetch all weights first, retrying on network drops
    if key in SMOKE_SKIP: continue
    for attempt in range(8):
        try:
            snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]); break
        except Exception as e:
            log(f"download {repo} failed ({type(e).__name__}); retry {attempt+1}/8 in 60 s"); time.sleep(60)
    else:
        raise SystemExit(f"could not download {repo}")
log("all models cached")
tok = AutoTokenizer.from_pretrained(M["target"]); tok.pad_token = tok.eos_token
neo_tok = AutoTokenizer.from_pretrained(M["neo125"])
assert neo_tok.get_vocab() == tok.get_vocab(), "GPT-Neo tokenizer differs from GPT-2's"


def clean(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([.,;:!?)\]'])", r"\1", s); s = re.sub(r"([(\[]) ", r"\1", s)
    return s.replace(" n't", "n't").replace(" 's", "'s")


lines = []
for split in ("test", "validation"):
    df = pd.read_parquet(hf_hub_download("Salesforce/wikitext",
                         f"wikitext-103-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset"))
    lines += [clean(l.strip()) for l in df.text if len(l.split()) > 40 and not l.strip().startswith("=")]
random.shuffle(lines)
# test prompts: the opening 16 words of the first N_TEST paragraphs.
# training prompts: non-overlapping 16-word windows from the OTHER paragraphs only,
# so no test paragraph's text ever reaches the distilled outsider.
test_p = [" ".join(l.split()[:16]) for l in lines[:N_TEST]]
train_p = []
for l in lines[N_TEST:]:
    w = l.split()
    train_p += [" ".join(w[o:o + 16]) for o in range(0, len(w) - 16, 24)]
random.shuffle(train_p); train_p = train_p[:N_TRAIN]
assert not set(test_p) & set(train_p) and len(train_p) == N_TRAIN, (len(train_p), N_TRAIN)


import pickle
CKPT = os.path.join(OUT, "b2_checkpoint.pkl")
state = pickle.load(open(CKPT, "rb")) if os.path.exists(CKPT) else {}
if state: log("resuming from checkpoint; done so far:", sorted(state))


def save(**kw):
    state.update(kw); tmp = CKPT + ".tmp"
    pickle.dump(state, open(tmp, "wb")); os.replace(tmp, CKPT)


def load(name, half=False):
    dt = torch.float16 if (half and DEV == "cuda") else torch.float32
    return AutoModelForCausalLM.from_pretrained(M[name], dtype=dt).to(DEV).eval()


def free():
    """Release GPU memory. Callers must `del` their model first: deleting a name
    inside this function would not drop the caller's reference."""
    import gc; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()


@torch.no_grad()
def next_logits(model, prefix):
    return model(torch.tensor([prefix], device=DEV)).logits[0, -1].float()


def sample(logits, g):
    v, i = torch.topk(logits, 50)
    return i[torch.multinomial(torch.softmax(v, -1).cpu(), 1, generator=g)].item()


# ---------------- build the self/other set ----------------
if "items" in state:
    items = state["items"]
else:
    log("building continuations and insertions")
    target = load("target"); g = torch.Generator().manual_seed(2)
    conts = []
    for prompt in test_p:
        ids, cont = tok(prompt)["input_ids"], []
        for _ in range(CONT):
            cont.append(sample(next_logits(target, ids + cont), g))
        conts.append((ids, cont))
    inserter = load("inserter", half=True)
    fluent = {(k, p): sample(next_logits(inserter, ids + cont[:p]), g)
              for k, (ids, cont) in enumerate(conts) for p in POS}
    del inserter; free()
    items = []
    for k, (ids, cont) in enumerate(conts):
        for p in POS:
            prefix, own = ids + cont[:p], cont[p]
            items.append(dict(k=k, prefix=prefix, tok=own, kind="self"))
            if fluent[(k, p)] != own:
                items.append(dict(k=k, prefix=prefix, tok=fluent[(k, p)], kind="fluent"))
            off = fluent[((k + 1) % N_TEST, p)]
            if off != own:
                items.append(dict(k=k, prefix=prefix, tok=off, kind="offctx"))
    save(items=items)
kinds = np.array([i["kind"] for i in items]); labels = (kinds == "self").astype(int)
groups = np.array([i["k"] for i in items])
log("items:", pd.Series(kinds).value_counts().to_dict())


@torch.no_grad()
def score_all(model):
    return np.array([torch.log_softmax(next_logits(model, it["prefix"]), -1)[it["tok"]].item()
                     for it in items])


scores = state.get("scores", {})
if ("target" not in scores or "train_txt" not in state) and "target" not in globals():
    target = load("target")
if "target" not in scores:
    scores["target"] = score_all(target); save(scores=scores)

# ---------------- distillation data: target outputs on held-out prompts ----------------
if "train_txt" in state:
    train_txt = state["train_txt"]
else:
    log("generating", N_TRAIN, "target continuations for distillation")
    train_txt = []; tok.padding_side = "left"
    with torch.no_grad():
        for b in range(0, N_TRAIN, 32):
            enc = tok(train_p[b:b + 32], return_tensors="pt", padding=True).to(DEV)
            gen = target.generate(**enc, max_new_tokens=48, do_sample=True, top_k=50,
                                  pad_token_id=tok.eos_token_id)
            for row, m in zip(gen.cpu(), enc["attention_mask"].cpu()):
                plen = int(m.sum()); full = row[len(m) - plen:].tolist()
                train_txt.append((full[:plen], full[plen:]))
    save(train_txt=train_txt)
if "target" in globals():
    del target; free()

for name, key, half in (("copy", "target", False), ("gpt2-small", "small", False),
                        ("neo-125m", "neo125", False)) + (() if SMOKE else (("neo-1.3b", "neo13", True),)):
    if name in scores: continue
    log("scoring", name); m = load(key, half); scores[name] = score_all(m); del m; free(); save(scores=scores)

# ---------------- distilled outsider: GPT-Neo 125M trained only on the target's outputs ----------------
need_dist = not all(f"neo-125m distilled {e}ep" in scores for e in (1, 3))
dist = AutoModelForCausalLM.from_pretrained(M["neo125"], dtype=torch.float32).to(DEV) if need_dist else None
opt = torch.optim.AdamW(dist.parameters(), lr=5e-5) if need_dist else None
for ep in range(3 if need_dist else 0):
    dist.train(); random.shuffle(train_txt)
    for b in range(0, len(train_txt), 8):
        batch = train_txt[b:b + 8]; L = max(len(p) + len(c) for p, c in batch)
        x = torch.full((len(batch), L), tok.eos_token_id); y = torch.full((len(batch), L), -100)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for r, (p, c) in enumerate(batch):
            s = p + c; x[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
            y[r, len(p):len(s)] = torch.tensor(c)
        loss = dist(input_ids=x.to(DEV), attention_mask=att.to(DEV), labels=y.to(DEV)).loss
        opt.zero_grad(); loss.backward(); opt.step()
    dist.eval(); log(f"distillation epoch {ep+1}, last loss {loss.item():.3f}")
    if ep + 1 in (1, 3):
        scores[f"neo-125m distilled {ep+1}ep"] = score_all(dist); save(scores=scores)
del dist; free()

# ---------------- statistics ----------------
def auroc(s, m): return roc_auc_score(labels[m], s[m])


rng = np.random.default_rng(0); ks = np.unique(groups)
boots = [np.concatenate([np.where(groups == q)[0] for q in rng.choice(ks, len(ks))]) for _ in range(2000)]
ref = scores["target"]; rows = []
for name, s in scores.items():
    row = dict(scorer=name, spearman_with_self=round(spearmanr(s, ref).correlation, 3))
    for kind in ("fluent", "offctx"):
        m = (kinds == "self") | (kinds == kind)
        est = auroc(s, m); d_est = auroc(ref, m) - est
        vals, diffs = [], []
        for idx in boots:
            idx = idx[m[idx]]
            if len(set(labels[idx])) < 2: continue
            a_s = roc_auc_score(labels[idx], s[idx]); a_r = roc_auc_score(labels[idx], ref[idx])
            vals.append(a_s); diffs.append(a_r - a_s)
        row[f"auroc_{kind}"] = f"{est:.3f} [{np.percentile(vals, 2.5):.3f}-{np.percentile(vals, 97.5):.3f}]"
        row[f"self_minus_this_{kind}"] = (f"{d_est:+.3f} [{np.percentile(diffs, 2.5):+.3f}"
                                          f" to {np.percentile(diffs, 97.5):+.3f}]")
    rows.append(row)
rep = pd.DataFrame(rows)
print("\n" + rep.to_string(index=False), flush=True)
np.savez(os.path.join(OUT, "b2_scores.npz"), labels=labels, kinds=kinds, groups=groups,
         **{re.sub(r"\W+", "_", k): v for k, v in scores.items()})
json.dump(dict(table=rows, counts=pd.Series(kinds).value_counts().to_dict(), device=DEV,
               n_test=N_TEST, n_train=N_TRAIN), open(os.path.join(OUT, "b2_results.json"), "w"), indent=1)
log("done")
