# Qubit-Efficient QUBO for Interbank Payment Settlement and Counterparty Netting

*Use case originally posed by Quantum Dice's Trinity 2026 Challenge; this implementation
and the qubit/QUBO approach are my own independent work.*

Encodes the interbank payment settlement and counterparty netting problem as a QUBO (Quadratic Unconstrained Binary Optimisation) and implements the IQP + Master-Satellite (IQPMS) decomposition from De Santis et al. (2026) to reduce slack variable overhead. The IQPMS investigation focuses on gridlock scenarios, where the IN/OUT constraint holds and achieves **84% fewer slack variables** (4 vs 25) compared to the standard squared-penalty encoding, verified by exhaustive enumeration.

For the problem background, see Section 1 of the notebook.

## Files

| File | Description |
|------|-------------|
| `qubo_interbank_settlement.ipynb` | Main notebook — QUBO formulation, verification, baselines, IQPMS investigation and implementation. All outputs embedded. |
| `report.pdf` | 2-page technical report summarising problem, methods, results. |
| `iqpms_qmatrix.py` | Supplementary: full IQPMS Q-matrix implementation with detailed polynomial verification (~1100 lines). Referenced in notebook Section 10. |
| `dataset_exp1.csv` | Experiment 1: 5 banks, 8 payments, 30% liquidity. |
| `dataset_exp2.csv` | Experiment 2: 6 banks, 12 payments, 35% liquidity. |
| `dataset_exp3.csv` | Experiment 3: 7 banks, 14 payments, 20% liquidity (severe gridlock). |
| `dataset_gridlock.csv` | IQPMS scenario: 6 banks, 9 payments, 25% liquidity (Section 10). |

## Requirements

Python 3.8+ with `numpy` and `scipy`. No other dependencies.

## Running

The notebook runs end-to-end in ~90 seconds. Section 10 (IQPMS) takes ~30s due to exhaustive verification over 8192 configurations. The supplementary script `iqpms_qmatrix.py` runs standalone: `python3 iqpms_qmatrix.py`.

## References

1. De Santis, D., Tirone, S., Marmi, S., & Giovannetti, V. (2026). *Optimized QUBO formulation methods for quantum computing.* Quantum Science and Technology, 11, 015056. Also available as arXiv:2406.07681v2.
2. Glover, F., Kochenberger, G., Hennig, R., & Du, Y. (2022). *Quantum bridge analytics I: a tutorial on formulating and using QUBO models.* Annals of Operations Research, 314(1), 141–183.
3. Lucas, A. (2014). *Ising formulations of many NP problems.* Frontiers in Physics, 2, 5.
4. Bech, M. L. & Soramäki, K. (2001). *Gridlock resolution in interbank payment systems.* Bank of Finland Discussion Papers.