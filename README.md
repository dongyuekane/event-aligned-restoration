# Event-aligned restoration scheduling

Research code for the noisy-diagnosis experiment in **Event-Aligned Restoration Scheduling with Diagnostic Uncertainty**, by Dongyue Kan. It tests when to wait, inspect, retry, or repair while a simulated restoration command may complete. Physical completion and acknowledgement occur at separate times. Diagnostic labels update a finite belief distribution with known error probabilities.

The experiment compares an offline-generated, frozen arithmetic policy, two- and three-step rollout, full belief-state dynamic programming, and two lower-bound pruning variants. Every method uses the same observation model and action menu. The bound search evaluates a fully observed relaxation and uses its expected cost to prune candidate actions.

## Run locally

Use Python 3.12. The experiments and result checks use the standard library. The frozen policy is evaluated as local Python code and runs offline.

```bash
python3 study.py smoke --output generated/smoke
python3 study.py development --output generated/replay
python3 study.py confirmation --output generated/replay
python3 study.py timing --output generated/replay
python3 verify.py --replay generated/replay
```

The smoke run evaluates two confirmation configurations. The full run replays 8 development and 24 confirmation configurations and takes three fresh-cache root timings per confirmation configuration. Run the commands serially on an idle machine for timing comparisons. `python3 verify.py` checks the archived measurements using only the included files.

## Archived confirmation results

| Method | Mean expected unserved energy (kWh) | Matches with full DP | Median root time (ms) |
| --- | ---: | ---: | ---: |
| Frozen arithmetic policy | 178.944 | 17/24 | 0.131 |
| Two-step rollout | 176.754 | 23/24 | 2.881 |
| Three-step rollout | 176.624 | 24/24 | 3.309 |
| Full dynamic programming | 176.624 | 24/24 | 25.711 |
| Root-only bounds | 176.624 | 24/24 | 19.140 |
| Recursive bounds | 176.624 | 24/24 | 5.464 |

Recursive pruning reduced archived exact-search time by 78.75%. Three-step rollout reached the same expected energy in all 24 configurations at the lowest measured time among the methods that matched full DP in every case. The root timings include model and bound setup and vary by machine. Full-policy expected energy comes from a separate recursive evaluation.

## Study files and scope

- `model.py` implements the synthetic two-block finite model, including outstanding command completion and acknowledgement.
- `observations.py` applies the likelihood update to repeated temporary/persistent diagnostic reports.
- `frozen_policy.py` contains the single archived arithmetic policy produced by offline language-model synthesis.
- `study.py` evaluates the six methods over frozen inputs. `verify.py` checks the archive and compares a fresh replay.
- `archive/` contains 32 synthetic timing/prior configurations, raw full-policy and root-timing rows, and the original summaries.

The configurations vary time and prior parameters for the same simulated action menu. Diagnostic error probabilities of 0.1, 0.2, and 0.3 are experiment inputs. This package covers the manuscript's noisy-diagnosis study. The manuscript also describes a separate exact-report experiment. The frozen policy artifact lets readers replay the study without an account or a remote service.
