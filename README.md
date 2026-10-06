# Eco-Friendly Daily Commute Helper

**A Quantitative, Scenario-Based Study, Simulated Analytical Demonstration, and AI-Driven Conceptual
Framework for Sustainable Urban Commuting**

Critical Research Methodology (CRM) paper — Chandigarh University.

## Important research-integrity notice

This repository contains a **proposed** quantitative research design and a **simulated** analytical
demonstration. No survey was administered, no participant was recruited, and no system was deployed.
Every numerical result in the paper was generated from a **synthetic dataset** produced by a computer
program with a fixed random seed. These numbers describe no real person and no real system, and they
must not be cited as empirical evidence about Chandigarh University students or about any deployed
commute assistant.

## Contents

| File | Description |
|---|---|
| `Eco_Friendly_Commute_Helper_Revised.pdf` | Complete revised paper (43 pages) |
| `Eco_Friendly_Commute_Helper_Revised.tex` | LaTeX source of the paper |
| `simulation_analysis.py` | Reproducible simulation + full statistical analysis (N = 120) |
| `make_figures.py` | Generates the seven paper figures |
| `simulated_dataset.csv` | The synthetic respondent-level dataset |
| `analysis_output.json` | Machine-readable statistical output |
| `fig_*.png` | Figures used in the paper |

## Reproducing the results

Requires Python 3 with `numpy`, `scipy`, `pandas`, `statsmodels`, `scikit-learn` and `matplotlib`.

```bash
python3 simulation_analysis.py   # writes simulated_dataset.csv and analysis_output.json
python3 make_figures.py          # writes fig_*.png
```

The generator uses the fixed seed `20251006`, so the dataset and every statistic are exactly
reproducible. Re-running `simulation_analysis.py` produces byte-identical output.

## Compiling the paper

```bash
pdflatex Eco_Friendly_Commute_Helper_Revised.tex
pdflatex Eco_Friendly_Commute_Helper_Revised.tex   # second pass for the table of contents
```
