"""
V4c (re-recognition / pratyabhijna) on the Spanda Self-Model.

A system and its twin (identical laws, different seed) each leave a 500-step history.
The system is then disrupted, relaxes for 500 steps, and must say which history was its own.

Disruption levels
  rings     - all ring phases scrambled (sd 1.5 rad); self-anchor untouched
  partial   - rings scrambled and anchor nudged (sd 0.05 rad)
  total     - rings scrambled and anchor reset to a random value

Recognisers
  self-raw     - compares its current anchor with each history's final anchor
  self-efcopy  - first undoes its own logged anchor updates (efference copy), then compares
  outsider     - sees ring phases only; infers each anchor from the phase dynamics
                 (V4b-style fit of the anchor force), then compares
"""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("vt", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "vimarsa_tests_lib.py"))
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
K, N, dt, C, wrap = vt.K, vt.N, vt.dt, vt.CENTER, vt.wrap
W_HIST = W_RELAX = 500
rng = np.random.default_rng(7)
JEFF = vt.JEFF


def known_force(theta):
    """Coupling part of the force (couplings known to the outsider from V4b, to 0.001)."""
    g = np.einsum("ij,kij->ki", JEFF, np.sin(theta[:, :, None] - theta[:, None, :]))
    for k in range(K):
        for l in range(K):
            if l != k:
                g[k] += vt.J_inter * np.sin(theta[k] - theta[l])
    return g


def run(theta, p, anchor, t, n):
    """Ground truth with Vimarsa loop; logs phases, anchors and own anchor updates."""
    phases, anchors, updates = [theta], [anchor], []
    for _ in range(n):
        theta, p = vt.step(theta, p, anchor)
        if t % vt.vimarsa_every == 0:
            d = vt.vimarsa_delta(theta, anchor)
            anchor = anchor + d
            updates.append(d)
        phases.append(theta); anchors.append(anchor)
        t += 1
    return theta, p, anchor, t, phases, anchors, updates


def infer_anchor(phases):
    """Outsider: fit gamma*sin(theta_Omega - a) from second differences of phases only."""
    rows, ys = [], []
    for i in range(0, len(phases) - 2, 2):
        a, b, c = phases[i], phases[i + 1], phases[i + 2]
        force = -(wrap(c - b) - wrap(b - a)) / dt ** 2          # = total force at b
        resid = force - known_force(b)
        for k in range(K):
            rows.append([np.sin(b[k, C]), np.cos(b[k, C])]); ys.append(resid[k, C])
    gc, gs = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)[0]
    return np.arctan2(-gs, gc)                                 # gc = g cos a, gs = -g sin a


def infer_anchor_with_rule(phases, t_start):
    """Outsider that has also learned the Vimarsa rule: read the anchor on the final
    constant-anchor segment, then run the rule backwards using the observed ring phases."""
    times = [t_start + i for i in range(len(phases) - 1)]
    upd = [i for i, t in enumerate(times) if t % vt.vimarsa_every == 0]   # step index of updates
    last = upd[-1] + 1                                                    # anchor constant after this
    a = infer_anchor(phases[last:])
    for i in reversed(upd):
        z = np.mean(np.exp(1j * phases[i + 1]))      # rule reads phases right after the step
        a_new = a
        for _ in range(50):                          # invert a_new = a + r|z|wrap(arg z - a)
            a = a_new - vt.vimarsa_rate * np.abs(z) * wrap(np.angle(z) - a)
    return a


def closer(x, a, b):
    """True if x is nearer to a than to b (on the circle)."""
    return abs(wrap(x - a)) < abs(wrap(x - b))


levels = ["rings", "partial", "total"]
RECS = ["self-raw", "self-efcopy", "outsider", "outsider+rule"]
acc = {lv: {r: [] for r in RECS} for lv in levels}
twin_gap = []

for seed in range(10):
    # self and twin: identical laws, different seeds; burn in, then record a history window
    sims = {}
    for who, s in (("self", seed), ("twin", seed + 100)):
        th, p, a = vt.init(s)
        th, p, a, t, *_ = run(th, p, a, 0, 2000)
        th, p, a, t, ph, an, _ = run(th, p, a, t, W_HIST)
        t_hist = t - W_HIST
        upd = [i for i in range(W_HIST) if (t_hist + i) % vt.vimarsa_every == 0]
        sims[who] = dict(th=th, p=p, a=a, t=t, ph=ph, a_end=an[-1],
                         a_hat=infer_anchor(ph),
                         a_hat_end=infer_anchor(ph[upd[-1] + 1:]))   # final constant segment
    S, T = sims["self"], sims["twin"]
    twin_gap.append(abs(wrap(S["a_end"] - T["a_end"])))

    for lv in levels:
        for _ in range(20):
            th = np.mod(S["th"] + rng.normal(0, 1.5, (K, N)), 2 * np.pi)
            a = S["a"]
            if lv == "partial":
                a = a + rng.normal(0, 0.05)
            elif lv == "total":
                a = rng.uniform(-np.pi, np.pi)
            th, p, a_now, t, ph, an, ups = run(th, S["p"], a, S["t"], W_RELAX)

            acc[lv]["self-raw"].append(closer(a_now, S["a_end"], T["a_end"]))
            a_back = a_now - sum(ups)                       # undo its own updates
            acc[lv]["self-efcopy"].append(closer(a_back, S["a_end"], T["a_end"]))
            acc[lv]["outsider"].append(closer(infer_anchor(ph), S["a_hat"], T["a_hat"]))
            acc[lv]["outsider+rule"].append(
                closer(infer_anchor_with_rule(ph, S["t"]), S["a_hat_end"], T["a_hat_end"]))

print(f"self-twin anchor gap after burn-in: median {np.median(twin_gap):.4f} rad "
      f"(range {min(twin_gap):.4f}-{max(twin_gap):.4f})")
print("\nRe-recognition accuracy (200 trials per level; 0.50 = chance)")
print(f"{'':10s}" + "".join(f"{r:>15s}" for r in RECS))
for lv in levels:
    print(f"{lv:10s}" + "".join(f"{np.mean(acc[lv][r]):15.2f}" for r in RECS))

# outsider calibration: how well does it read an anchor from phases alone?
errs = []
for seed in range(10):
    th, p, a = vt.init(seed)
    th, p, a, t, *_ = run(th, p, a, 0, 2000)
    th, p, a, t, ph, an, _ = run(th, p, a, t, W_HIST)
    errs.append(abs(wrap(infer_anchor(ph) - np.mean(an))))
print(f"\noutsider anchor-reading error at rest: median {np.median(errs):.2e} rad")

# per-trial agreement between self-efcopy and outsider+rule
agree = [np.mean(np.array(acc[lv]["self-efcopy"]) == np.array(acc[lv]["outsider+rule"])) for lv in levels]
print("per-trial agreement, self-efcopy vs outsider+rule:", dict(zip(levels, np.round(agree, 3))))
