"""
Q1 positive-control run: V1-V3 vimarsa-candidate tests on the Spanda Self-Model (L0 + L1).
Spec: spanda-self-model skill (validated parameters, + sign gradient, Stormer-Verlet).
Variants:
  full     - Vimarsa loop reads own rings (the model as built)
  cut      - Vimarsa loop disabled (anchor frozen)            [negative control]
  decoy    - Vimarsa loop reads an independent decoy system   [negative control]
  efcopy   - full + efference copy of own anchor updates      [V3 probe]
"""
import numpy as np

N, K, dt, CENTER = 7, 3, 0.01, 3
J_intra, J_inter, gamma = 0.3, 0.4, 0.8
vimarsa_rate, vimarsa_every = 0.02, 50
symbols = ["A", "A", "A", "Ω", "A", "A", "A"]
comp = {("A", "A"): 1.0, ("A", "Ω"): 0.2, ("Ω", "A"): 0.2, ("Ω", "Ω"): 2.0}
JEFF = np.array([[J_intra + comp[(symbols[i], symbols[j])] for j in range(N)] for i in range(N)])
np.fill_diagonal(JEFF, 0.0)


def dH(theta, anchor):
    diff = theta[:, :, None] - theta[:, None, :]            # [k,i,j]
    g = np.einsum("ij,kij->ki", JEFF, np.sin(diff))          # +J sin (intra)
    for k in range(K):
        for l in range(K):
            if l != k:
                g[k] += J_inter * np.sin(theta[k] - theta[l])  # +J sin (inter)
    g[:, CENTER] += gamma * np.sin(theta[:, CENTER] - anchor)  # +gamma sin (anchor)
    return g


def step(theta, p, anchor):
    p_h = p - dH(theta, anchor) * dt / 2
    theta = np.mod(theta + p_h * dt, 2 * np.pi)
    p = p_h - dH(theta, anchor) * dt / 2
    return theta, p


def mean_phase(theta):
    return np.angle(np.mean(np.exp(1j * theta)))


def vimarsa_delta(observed_theta, anchor):
    z = np.mean(np.exp(1j * observed_theta))
    return vimarsa_rate * np.abs(z) * np.angle(np.exp(1j * (np.angle(z) - anchor)))


def wrap(x):
    return np.angle(np.exp(1j * x))


def init(seed):
    rng = np.random.default_rng(seed)
    base = np.array([0.5, 0.52, 0.53, 0.56, 0.53, 0.52, 0.5]) * 2 * np.pi
    theta = np.array([base + rng.normal(0, 0.1, N) for _ in range(K)])
    return theta, np.zeros((K, N)), np.pi


class Sim:
    def __init__(self, seed, mode):
        self.mode = mode
        self.theta, self.p, self.anchor = init(seed)
        self.decoy_theta, self.decoy_p, _ = init(seed + 10_000)
        self.decoy_theta = np.mod(self.decoy_theta + 1.3, 2 * np.pi)  # decoy sits elsewhere
        self.efcopy = self.anchor   # predicted anchor = last value the system itself set
        self.t = 0

    def run(self, n):
        for _ in range(n):
            self.theta, self.p = step(self.theta, self.p, self.anchor)
            if self.mode == "decoy":
                self.decoy_theta, self.decoy_p = step(self.decoy_theta, self.decoy_p, np.pi)
            if self.t % vimarsa_every == 0 and self.mode != "cut":
                src = self.decoy_theta if self.mode == "decoy" else self.theta
                self.anchor = self.anchor + vimarsa_delta(src, self.anchor)
                self.efcopy = self.anchor
            self.t += 1
        return self

    def copy(self):
        import copy
        return copy.deepcopy(self)


BURN, POST, SEEDS, DELTA = 2000, 3000, range(10), 0.5
modes = ["full", "cut", "decoy"]
res = {m: {"V1a": [], "V1b": []} for m in modes}

for m in modes:
    for s in SEEDS:
        base = Sim(s, m).run(BURN)

        # V1a encoding: rotate the system's own rings by DELTA; does the self-state follow?
        a = base.copy(); a.theta = np.mod(a.theta + DELTA, 2 * np.pi)
        a0 = a.anchor; a.run(POST)
        c = base.copy(); c.run(POST)
        res[m]["V1a"].append(wrap(a.anchor - c.anchor) / DELTA)   # 1 = fully tracks own state

        # V1b causal: intervene on the self-state; do the rings move?
        b = base.copy(); b.anchor += DELTA; b.efcopy = b.anchor; b.run(POST)
        res[m]["V1b"].append(wrap(mean_phase(b.theta) - mean_phase(c.theta)) / DELTA)

# Energy sanity check on the full model (no Vimarsa updates during the check window)
th, p, anc = init(0)
def H(th, p, anc):
    diff = th[:, :, None] - th[:, None, :]
    V = -0.5 * np.einsum("ij,kij->", JEFF, np.cos(diff))
    V -= 0.5 * sum(J_inter * np.cos(th[k] - th[l]).sum() for k in range(K) for l in range(K) if l != k)
    V -= gamma * np.cos(th[:, CENTER] - anc).sum()
    return V + 0.5 * (p ** 2).sum()
E0 = H(th, p, anc); drift = 0.0
for _ in range(3000):
    th, p = step(th, p, anc); drift = max(drift, abs(H(th, p, anc) - E0))

# V3 self/other discrimination, matched design:
# at a Vimarsa update step, branch S lets the system update its own anchor;
# branch X suppresses the update and an experimenter writes the SAME anchor value.
v3 = {"no_efcopy_maxdiff": [], "efcopy_err_self": [], "efcopy_err_ext": []}
for s in SEEDS:
    base = Sim(s, "full").run(BURN)
    while base.t % vimarsa_every != 0:
        base.run(1)
    d = vimarsa_delta(base.theta, base.anchor)
    S = base.copy(); X = base.copy()
    S.anchor = S.anchor + d; S.efcopy = S.anchor            # self-generated
    X.anchor = X.anchor + d                                 # externally injected, efcopy untouched
    v3["efcopy_err_self"].append(abs(wrap(S.anchor - S.efcopy)))
    v3["efcopy_err_ext"].append(abs(wrap(X.anchor - X.efcopy)))
    # observable state without efference copy: (theta, p, anchor)
    S.t += 1; X.t += 1
    S.run(500); X.run(500)
    v3["no_efcopy_maxdiff"].append(max(np.abs(S.theta - X.theta).max(),
                                       np.abs(S.p - X.p).max(), abs(S.anchor - X.anchor)))

print(f"energy drift (3000 steps, loop off): {drift:.2e}")
for m in modes:
    v1a, v1b = np.array(res[m]["V1a"]), np.array(res[m]["V1b"])
    print(f"{m:6s} V1a tracking = {v1a.mean():+.3f} ± {v1a.std():.3f} | "
          f"V1b causal = {v1b.mean():+.3f} ± {v1b.std():.3f}")
print("V3 without efference copy: max state difference self vs external =",
      f"{max(v3['no_efcopy_maxdiff']):.2e}")
print("V3 with efference copy: prediction error self =",
      f"{np.mean(v3['efcopy_err_self']):.2e}, external = {np.mean(v3['efcopy_err_ext']):.2e}",
      f"(min external {np.min(v3['efcopy_err_ext']):.2e})")
