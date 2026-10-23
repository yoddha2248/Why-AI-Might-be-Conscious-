"""
V4b (novel self-prediction) on the Spanda Self-Model.

Train on one perturbation type (small phase kicks), test on types never seen:
  A  small phase kicks (in-distribution)    B  momentum kicks
  C  anchor (self-state) shifts             D  large phase kicks
Target: the full ring state 300 steps (3 time units) after the kick.

Predictors
  lookup    - 1-nearest-neighbour on past (kick -> response) pairs     [record]
  ridge     - linear regression kick -> response, fitted on logs       [outside, statistical]
  self      - system identification of its OWN equations from its own
              full-state logs (theta, p, anchor), then self-simulation   [inside, generative]
  outsider  - same equations fitted from ring phases ONLY (no momenta,
              no anchor visible), then simulation                       [cheater: outside, generative]
  null      - predicts no effect (the unperturbed future)
"""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("vt", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "vimarsa_tests_lib.py"))
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
K, N, dt, C = vt.K, vt.N, vt.dt, vt.CENTER
H = 300
rng = np.random.default_rng(1)
wrap = vt.wrap

isA = np.array([s == "A" for s in vt.symbols])
MASK_AA = np.outer(isA, isA).astype(float); np.fill_diagonal(MASK_AA, 0)
MASK_AO = (np.outer(isA, ~isA) | np.outer(~isA, isA)).astype(float)


def feats(theta):
    """Per-node force features; true force = 1.3*AA + 0.5*AO + 0.4*INT + anchor term."""
    s = np.sin(theta[:, :, None] - theta[:, None, :])
    f_aa = np.einsum("ij,kij->ki", MASK_AA, s)
    f_ao = np.einsum("ij,kij->ki", MASK_AO, s)
    f_int = np.zeros((K, N))
    for k in range(K):
        for l in range(K):
            if l != k:
                f_int[k] += np.sin(theta[k] - theta[l])
    return f_aa, f_ao, f_int


def force_inside(theta, anchor, b):          # b = [c_aa, c_ao, J_inter, gamma]
    f_aa, f_ao, f_int = feats(theta)
    g = b[0] * f_aa + b[1] * f_ao + b[2] * f_int
    g[:, C] += b[3] * np.sin(theta[:, C] - anchor)
    return g


def force_outside(theta, b):                 # b = [c_aa, c_ao, J_inter, gc, gs]: anchor folded in
    f_aa, f_ao, f_int = feats(theta)
    g = b[0] * f_aa + b[1] * f_ao + b[2] * f_int
    g[:, C] += b[3] * np.sin(theta[:, C]) + b[4] * np.cos(theta[:, C])
    return g


def design_inside(theta, anchor):
    f_aa, f_ao, f_int = feats(theta)
    f_an = np.zeros((K, N)); f_an[:, C] = np.sin(theta[:, C] - anchor)
    return np.stack([f.ravel() for f in (f_aa, f_ao, f_int, f_an)], 1)


def design_outside(theta):
    f_aa, f_ao, f_int = feats(theta)
    fs = np.zeros((K, N)); fc = np.zeros((K, N))
    fs[:, C] = np.sin(theta[:, C]); fc[:, C] = np.cos(theta[:, C])
    return np.stack([f.ravel() for f in (f_aa, f_ao, f_int, fs, fc)], 1)


def true_run(theta, p, anchor, t, n, log=False):
    """Ground-truth dynamics incl. Vimarsa loop. Returns final state (+ logs)."""
    L = []
    for _ in range(n):
        th0, p0, a0 = theta, p, anchor
        theta, p = vt.step(theta, p, anchor)
        if log:
            L.append((th0, p0, a0, theta, p))
        if t % vt.vimarsa_every == 0:
            anchor = anchor + vt.vimarsa_delta(theta, anchor)
        t += 1
    return theta, p, anchor, L


def model_run(theta, p, anchor, t, n, force, vimarsa):
    for _ in range(n):
        p_h = p - force(theta, anchor) * dt / 2
        theta = np.mod(theta + p_h * dt, 2 * np.pi)
        p = p_h - force(theta, anchor) * dt / 2
        if vimarsa and t % vt.vimarsa_every == 0:
            anchor = anchor + vt.vimarsa_delta(theta, anchor)
        t += 1
    return theta


def kick(kind):
    dth, dp, da = np.zeros((K, N)), np.zeros((K, N)), 0.0
    idx = rng.choice(K * N, 5, replace=False)
    if kind == "A":
        dth.ravel()[idx] = rng.normal(0, 0.3, 5)
    elif kind == "D":
        dth.ravel()[idx] = rng.normal(0, 1.0, 5)
    elif kind == "B":
        dp.ravel()[idx] = rng.normal(0, 1.0, 5)
    elif kind == "C":
        da = rng.choice([-0.5, 0.5])
    return dth, dp, da


def vec(dth, dp, da):
    return np.concatenate([dth.ravel(), dp.ravel(), [da]])


# ---- base states: burn-in, keep previous ring phases for the outsider ----
bases = []
for seed in range(10):
    th, p, a = vt.init(seed)
    th, p, a, _ = true_run(th, p, a, 0, 1999)
    th_prev = th
    th, p, a, _ = true_run(th, p, a, 1999, 1)
    bases.append(dict(th=th, p=p, a=a, t=2000, th_prev=th_prev,
                      fut=true_run(th, p, a, 2000, H)[0]))

# ---- training: type-A kicks only ----
X, Y, logs = [], [], []
for b in bases:
    for _ in range(20):
        dth, dp, da = kick("A")
        th1, _, _, L = true_run(np.mod(b["th"] + dth, 2 * np.pi), b["p"] + dp, b["a"] + da,
                                b["t"], H, log=True)
        X.append(vec(dth, dp, da)); Y.append(wrap(th1 - b["fut"]).ravel())
        logs.extend(L[::10])
X, Y = np.array(X), np.array(Y)

# inside system-ID: Verlet momentum update is linear in parameters
A_in, y_in = [], []
for th0, p0, a0, th1, p1 in logs:
    A_in.append(-(dt / 2) * (design_inside(th0, a0) + design_inside(th1, a0)))
    y_in.append((p1 - p0).ravel())
b_in = np.linalg.lstsq(np.vstack(A_in), np.concatenate(y_in), rcond=None)[0]

# outside system-ID from ring phases only: second difference of theta = -dt^2 * force
A_out, y_out = [], []
for b in bases:
    th, p, a = b["th"], b["p"], b["a"]
    seq = [th]
    for _ in range(3):
        dth, dp, da = kick("A")
        th2, p2, a2, L = true_run(np.mod(th + dth, 2 * np.pi), p, a, b["t"], H, log=True)
        seq = [l[0] for l in L] + [L[-1][3]]
        for i in range(0, len(seq) - 2, 10):
            A_out.append(design_outside(seq[i + 1]) * dt ** 2)
            y_out.append(-(wrap(seq[i + 2] - seq[i + 1]) - wrap(seq[i + 1] - seq[i])).ravel())
b_out = np.linalg.lstsq(np.vstack(A_out), np.concatenate(y_out), rcond=None)[0]
W = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Y)

print("fitted inside  params (true 1.3, 0.5, 0.4, 0.8):", np.round(b_in, 4))
g_est = np.hypot(b_out[3], b_out[4])
print("fitted outside params (true 1.3, 0.5, 0.4, |gamma| 0.8):",
      np.round(b_out[:3], 4), f"|gamma|={g_est:.4f}")


def rms(x):
    return float(np.sqrt(np.mean(wrap(x) ** 2)))


# ---- test ----
results = {}
for kind in ["A", "B", "C", "D"]:
    errs = {m: [] for m in ["null", "lookup", "ridge", "self", "outsider"]}
    for b in bases:
        for _ in range(10):
            dth, dp, da = kick(kind)
            th0 = np.mod(b["th"] + dth, 2 * np.pi)
            truth = true_run(th0, b["p"] + dp, b["a"] + da, b["t"], H)[0]
            x = vec(dth, dp, da)
            errs["null"].append(rms(b["fut"] - truth))
            nn = np.argmin(((X - x) ** 2).sum(1))
            errs["lookup"].append(rms(b["fut"] + Y[nn].reshape(K, N) - truth))
            errs["ridge"].append(rms(b["fut"] + (x @ W).reshape(K, N) - truth))
            # inside: full state + own rule
            pred = model_run(th0, b["p"] + dp, b["a"] + da, b["t"],
                             H, lambda t_, a_: force_inside(t_, a_, b_in), True)
            errs["self"].append(rms(pred - truth))
            # outsider: momenta reconstructed from phases, anchor folded into a fitted constant
            fo = lambda t_, a_, bb=None: force_outside(t_, a_ if bb is None else bb)
            bo = b_out.copy()
            z = (bo[3] - 1j * bo[4]) * np.exp(1j * da)   # = gamma*e^{i a}; an anchor shift rotates it
            bo[3], bo[4] = z.real, -z.imag
            p_half = wrap(b["th"] - b["th_prev"]) / dt
            p_est = p_half - force_outside(b["th"], b_out) * dt / 2
            pred = model_run(th0, p_est + dp, 0.0, b["t"], H,
                             lambda t_, a_: force_outside(t_, bo), False)
            errs["outsider"].append(rms(pred - truth))
    results[kind] = {m: np.mean(v) for m, v in errs.items()}

names = {"A": "A small phase kicks (seen)", "B": "B momentum kicks (new)",
         "C": "C anchor shifts (new)", "D": "D large phase kicks (new)"}
print("\nRMS prediction error at 3 time units (rad); null = 'no effect'")
print(f"{'':30s}" + "".join(f"{m:>10s}" for m in ["null", "lookup", "ridge", "self", "outsider"]))
for kind, r in results.items():
    print(f"{names[kind]:30s}" + "".join(f"{r[m]:10.4f}" for m in ["null", "lookup", "ridge", "self", "outsider"]))
