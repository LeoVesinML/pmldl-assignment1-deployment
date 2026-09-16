# Dataset — UCI Wine Quality

* **Source:** [UCI Machine Learning Repository, dataset 186](https://archive.ics.uci.edu/dataset/186/wine+quality)
* **Reference:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis.
  *Modeling wine preferences by data mining from physicochemical properties.*
  Decision Support Systems, 47(4):547–553, 2009.
* **Licence:** CC BY 4.0
* **Files:** `data/raw/winequality-red.csv` (1 599 rows), `data/raw/winequality-white.csv` (4 898 rows)
* **Format:** `;`-separated CSV, one header row

The two files describe red and white *Vinho Verde* wines from the north of Portugal. Each row is
one wine, described by 11 physicochemical laboratory measurements plus a quality score that is the
median of at least three blind evaluations by wine experts (0 = very bad, 10 = excellent).

| Column | Unit | Meaning |
| --- | --- | --- |
| `fixed acidity` | g(tartaric acid)/dm³ | Non-volatile acids |
| `volatile acidity` | g(acetic acid)/dm³ | Vinegar taste when too high |
| `citric acid` | g/dm³ | Adds freshness |
| `residual sugar` | g/dm³ | Sugar left after fermentation |
| `chlorides` | g(sodium chloride)/dm³ | Saltiness |
| `free sulfur dioxide` | mg/dm³ | SO₂ still active as a preservative |
| `total sulfur dioxide` | mg/dm³ | Free + bound SO₂ |
| `density` | g/cm³ | Close to water, driven by alcohol and sugar |
| `pH` | – | Acidity on a 0–14 scale (most wines: 3–4) |
| `sulphates` | g(potassium sulphate)/dm³ | Antimicrobial additive |
| `alcohol` | % vol | Alcoholic strength |
| `quality` | score 0–10 | **Label source** |

## How the pipeline uses it

* The `wine_type` column (`red` / `white`) is added when the two files are concatenated and is used
  as a categorical feature.
* The label is binarised: `is_good_quality = quality >= 6` (see `params.yaml → data.good_quality_threshold`),
  which yields a ≈63 / 37 class split.
* Known data-quality issues handled by stage 1: ~1 177 exact duplicate rows and a long right tail in
  `residual sugar`, `chlorides` and `free sulfur dioxide` (handled with a 3·IQR fence).
  The published files contain no missing values, but the imputation step is implemented and tested
  so the pipeline stays correct if an incomplete data drop ever arrives.
