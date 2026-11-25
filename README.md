# Fractional Power-Law Attention in Generalized Morrey Spaces

Code accompanying the paper

> Süleyman Çelik, *Fractional Power-Law Attention as a Nonlinear Riesz Potential: Boundedness, Stability and Long-Context Behaviour in Generalized Morrey Spaces*, submitted to *Neural Networks* (2026).

## Contents

| File | Description |
| --- | --- |
| `deney.py` | Experiments 1–2 and the check of Theorem 1: critical exponent of the long-context growth law, the L² critical case, and attention dilution. NumPy only. Runs in about 5 seconds on a CPU. |
| `deney3.py` | Experiment 3: length generalization of small Transformers (softmax, RoPE, ALiBi, fractional attention with fixed and learnable order) on a long-range "most recent value" recall task. PyTorch. |
| `Deney3_Kesirli_Attention_Colab.ipynb` | Google Colab notebook for Experiment 3 (same code as `deney3.py`). Tested on a T4 GPU. |
| `deney3_sonuclar.json` | Raw results of Experiment 3 (3 seeds). |

## Requirements

- Python 3.10 or newer
- NumPy (all experiments)
- PyTorch 2.x (Experiment 3; a GPU is recommended)

```bash
pip install numpy torch
```

## Usage

Experiments 1–2 (results are written to `sonuclar.json`):

```bash
python deney.py
```

Experiment 3, quick smoke test on CPU (about 1 minute):

```bash
python deney3.py --hizli
```

Experiment 3, full run (7 models × 3 seeds; about 30 minutes on a T4 GPU):

```bash
python deney3.py
```

Alternatively, open `Deney3_Kesirli_Attention_Colab.ipynb` in Google Colab, set the runtime type to GPU (T4), and run all cells.

## Notes

Variable names and comments in the code are partly in Turkish (for example, *deney* = experiment, *sonuçlar* = results, *hızlı* = quick).

## License

MIT. See `LICENSE`.

## Contact

Süleyman Çelik, Fırat University, Elazığ, Türkiye. suleyman.celik@firat.edu.tr. ORCID: [0000-0003-3255-1950](https://orcid.org/0000-0003-3255-1950)

| `deney3_rope_sonuclar.json` | Raw results of the RoPE baseline in Experiment 3 (3 seeds, run on CPU with identical settings). |
