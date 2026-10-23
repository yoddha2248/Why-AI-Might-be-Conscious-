"""
V4a (distributed self) on the Spanda Self-Model, with two red-team cheaters.

Self/other events come from the real oscillator run: at each Vimarsa update the
anchor moves by d; with prob 1/2 the system did it (self, record updated), else an
experimenter wrote the same value (external, record untouched).

The record of "the last anchor value I set" is stored four ways across the 21 nodes:
  none        - no record (the model as built)
  single      - one node holds it (the V3 efference copy)
  replicated  - every node holds a copy            [cheater 1: redundancy]
  shamir      - 3-of-21 Shamir secret shares       [cheater 2: synergy + robustness]
Then a fraction k of nodes is ablated and a decoder tries to classify the event.
Also measured: can ANY single surviving node classify alone? (redundant vs synergistic)
"""
import importlib.util, random, sys
import numpy as np

spec = importlib.util.spec_from_file_location("vt", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "vimarsa_tests_lib.py"))
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)

P = 2**61 - 1          # prime field for Shamir
T = 3                  # threshold
NODES = vt.K * vt.N    # 21
SCALE = 10**9
rng = random.Random(0)


def q(x):  # quantize an angle to the field
    return int(round((x % (2 * np.pi)) * SCALE)) % P


def shamir_split(secret):
    coeffs = [secret] + [rng.randrange(P) for _ in range(T - 1)]
    return {i: sum(c * pow(i, e, P) for e, c in enumerate(coeffs)) % P for i in range(1, NODES + 1)}


def shamir_join(shares):
    pts = list(shares.items())[:T]
    s = 0
    for i, (xi, yi) in enumerate(pts):
        num, den = 1, 1
        for j, (xj, _) in enumerate(pts):
            if i != j:
                num = num * (-xj) % P
                den = den * (xi - xj) % P
        s = (s + yi * num * pow(den, P - 2, P)) % P
    return s


# ---- generate matched self/external events from the oscillator ----
events = []  # (anchor_value_now, recorded_value, is_self)
for seed in range(10):
    sim = vt.Sim(seed, "full").run(vt.BURN)
    record = sim.anchor
    for _ in range(40):
        while sim.t % vt.vimarsa_every != 0:
            sim.theta, sim.p = vt.step(sim.theta, sim.p, sim.anchor); sim.t += 1
        d = vt.vimarsa_delta(sim.theta, sim.anchor)
        is_self = rng.random() < 0.5
        sim.anchor = sim.anchor + d
        if is_self:
            record = sim.anchor
        events.append((sim.anchor, record, is_self))
        sim.theta, sim.p = vt.step(sim.theta, sim.p, sim.anchor); sim.t += 1
        record = sim.anchor  # next event starts from a known-consistent record
events = events
print(f"events: {len(events)} ({sum(e[2] for e in events)} self)")


def store(scheme, rec):
    if scheme == "none":
        return {}
    if scheme == "single":
        return {7: q(rec)}                       # one designated node
    if scheme == "replicated":
        return {i: q(rec) for i in range(1, NODES + 1)}
    if scheme == "shamir":
        return shamir_split(q(rec))


def decode(scheme, surviving, anchor):
    """Return True (self), False (external), or None (cannot tell -> coin flip)."""
    if scheme in ("single", "replicated"):
        if not surviving:
            return None
        v = next(iter(surviving.values()))
    elif scheme == "shamir":
        if len(surviving) < T:
            return None
        v = shamir_join(surviving)
    else:
        return None
    return v == q(anchor)


def accuracy(scheme, k, trials=3):
    hits = n = 0
    for anchor, rec, is_self in events:
        shares = store(scheme, rec)
        for _ in range(trials):
            alive = set(rng.sample(range(1, NODES + 1), NODES - round(k * NODES)))
            guess = decode(scheme, {i: v for i, v in shares.items() if i in alive}, anchor)
            if guess is None:
                guess = rng.random() < 0.5
            hits += guess == is_self; n += 1
    return hits / n


def single_unit_accuracy(scheme):
    """Best accuracy any ONE node achieves alone (redundant info -> 1, synergistic -> 0.5)."""
    best = 0.5
    for node in range(1, NODES + 1):
        hits = 0
        for anchor, rec, is_self in events:
            shares = store(scheme, rec)
            sub = {node: shares[node]} if node in shares else {}
            if scheme == "shamir":
                g = None   # one share is uniform noise over the field: no decoder can use it
            else:
                g = decode(scheme, sub, anchor)
            if g is None:
                g = rng.random() < 0.5
            hits += g == is_self
        best = max(best, hits / len(events))
    return best


ks = [0.0, 0.1, 0.25, 0.5, 0.75, 0.86, 0.9]
schemes = ["none", "single", "replicated", "shamir"]
print("ablated fraction k: " + "  ".join(f"{k:>5.2f}" for k in ks))
for s in schemes:
    print(f"{s:10s}        " + "  ".join(f"{accuracy(s, k):5.2f}" for k in ks))
print("\nbest single-node accuracy (redundancy check):")
for s in schemes:
    print(f"  {s:10s} {single_unit_accuracy(s):.2f}")
