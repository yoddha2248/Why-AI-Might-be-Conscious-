# Why AI Might Be Conscious — and Why No Test Can Yet Tell

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23589039.svg)](https://doi.org/10.5281/zenodo.23589039)

**Code, results and essay: stress-testing self-model indicators of consciousness with cheats and outside observers.**

Proposed indicators of consciousness in AI often include self-related capacities: monitoring
one's own state, telling self-caused changes from external ones, predicting oneself,
recognising one's own past. This repository tests whether such indicators can tell a system
that has these capacities from one that only imitates them. For every test it builds the
cheapest cheat it can find and an outside observer, then asks whether either passes too.

Read the [essay](essay/essay.md) (also as [PDF](essay/Why_AI_Might_Be_Conscious.pdf)). It is a response to Seth, Block & Tononi, "Three reasons why AI is not
conscious and perhaps never will be", *Nature* 658, 866–868 (2026).

## Findings

1. **Every self-test on a minimal self-model was passed by a cheat or an outsider.**
   The testbed is a 21-oscillator system with an explicit self-variable (`spanda_testbed/`).
   - Telling self-caused from external changes: passed by one added variable, a log of
     the system's own updates.
   - Doing so robustly, with no single part holding the information: passed by copying
     that log to every unit, and by a 3-of-21 secret-sharing code that survives 86% of
     units being destroyed.
   - Predicting itself under kinds of disturbance never seen before: an outsider that saw
     only part of the state recovered the system's own equations and came within 0.5–4.6%
     of null error.
   - Recognising its own past against a near-identical twin: an outsider that learned the
     update rule agreed with the system's own answer on all 600 trials.
2. **A transformer keeps no record of authorship.** Feeding a GPT-2 Medium model its own
   tokens one at a time gives bit-identical activations to having generated them. Its
   only way to tell its own words from inserted ones is self-prediction ("would I have
   said this?").
3. **An exact copy matches the model; outsiders without access trail it slightly.**
   Detecting fluent inserted tokens (AUROC, 150 prompts, 2,000 paired bootstrap resamples):

   | Scorer | Access | AUROC | Self minus this (95% CI) |
   | --- | --- | --- | --- |
   | Target, GPT-2 Medium | — | 0.637 | — |
   | Exact copy | Full | 0.637 | 0 |
   | GPT-2 Small | None, shared training data | 0.606 | +0.031 [+0.017, +0.045] |
   | GPT-Neo 125M | None, different training data | 0.594 | +0.043 [+0.027, +0.059] |
   | GPT-Neo 1.3B | None, different training data | 0.588 | +0.049 [+0.034, +0.063] |
   | GPT-Neo 125M, distilled on the target's outputs (3 epochs) | Outputs only | 0.577 | +0.059 [+0.040, +0.079] |

   The outsiders' judgments still correlate with the target's at 0.94–0.98 (Spearman). The
   self's small edge is consistent with the privileged-access results of Binder et al.
   (2025) at frontier scale.

**Summary.** Each self-related indicator tested here either can be gamed (a record or an
outside model suffices) or reduces to privileged access, which is real but graded and
depends on what the outsider can see. Neither kind, on its own, establishes or rules out
experience.

**Limits.** One deterministic toy model; small language models only; the tests and cheats
are ours, and a different kind of test might resist them. Finding one would be the most
useful thing anyone could do with this code.

## Run it

Testbed (Python 3.9+, NumPy only; about 75 seconds in total on a laptop CPU):

```
pip install -r requirements-testbed.txt
python spanda_testbed/vimarsa_tests.py   # closed loop and self vs. other, with controls
python spanda_testbed/v4a_tests.py       # distributed self vs. copies and secret sharing
python spanda_testbed/v4b_tests.py       # novel self-prediction vs. an outside observer
python spanda_testbed/v4c_tests.py       # recognising itself against a twin
```

Transformer experiments (install PyTorch for your hardware first):

```
pip install -r requirements-transformer.txt
python transformer_pilot/b1_pilot.py     # CPU, about 55 minutes
python transformer_pilot/b2_gpu.py       # GPU with 4 GB is enough, about 20 minutes of compute
```

`b2_gpu.py` downloads about 11 GB of model weights into `.hf_cache/` and saves a
checkpoint after each stage, so an interrupted run resumes where it stopped. Set
`B2_SMOKE=1` for a quick pipeline check. All scripts are seeded; the outputs in
`results/` are those reported above.

## Repository layout

| Path | Contents |
| --- | --- |
| `spanda_testbed/` | The self-model (`vimarsa_tests_lib.py`) and the four test scripts |
| `transformer_pilot/` | B1 (GPT-2 family, CPU) and B2 (adds GPT-Neo outsiders, GPU) |
| `results/` | Outputs of every run, including per-token scores for B2 (`b2_scores.npz`) |
| `essay/` | The accompanying essay (Markdown, PDF and Word) |

The model's component names (anchor, vimarśa loop) come from Kashmir Śaiva philosophy,
which informed its design; the essay discusses that background. Nothing in the code or
results depends on it.

## References

- Seth, A. K., Block, N. & Tononi, G. (2026). Three reasons why AI is not conscious and perhaps never will be. *Nature* 658, 866–868.
- Butlin, P. et al. (2025). Identifying indicators of consciousness in AI systems. *Trends in Cognitive Sciences*.
- Birch, J. (2024). *The Edge of Sentience*, ch. "Large Language Models and the Gaming Problem". Oxford University Press.
- Binder, F. J. et al. (2025). Looking Inward: Language Models Can Learn About Themselves by Introspection. ICLR 2025. arXiv:2410.13787.
- Anthropic (2025). Emergent introspective awareness in large language models.

## Disclosure

Developed with Claude (Anthropic), which helped design the experiments, write the code
and draft this text. Anthropic's research is among the work discussed. The author is
responsible for the content.

## License and citation

Code: MIT (see `LICENSE`). The essay in `essay/` is © Ajay, 2026, and is not covered by the MIT licence.
To cite, use DOI [10.5281/zenodo.23589039](https://doi.org/10.5281/zenodo.23589039) or see `CITATION.cff`.
