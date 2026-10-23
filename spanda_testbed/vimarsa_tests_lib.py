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
