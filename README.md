# UK House Price Prediction

Machine learning project predicting average UK house prices (2015–2025) across 12 regions and 4 property types using scikit-learn regression models.

---

## Dataset

**File:** `uk_house_prices_raw.csv` &nbsp;|&nbsp; **Rows:** 6,336 &nbsp;|&nbsp; **Years:** 2015–2025

| Column | Description |
|--------|-------------|
| `Date` | Year-Month (YYYY-MM) |
| `Year` / `Month` | Temporal features |
| `Region` | 12 UK regions (London, South East, Scotland, etc.) |
| `PropertyType` | Detached, Semi-Detached, Terraced, Flat |
| `AveragePrice_GBP` | **Target variable** |
| `SalesVolume` | Number of sales transactions |
| `MedianPrice_GBP` | Median sale price |
| `PricePerSqM_GBP` | Price per square metre |

---

## Models & Results

Temporal train/test split — last 18 months held out as test set.

| Model | MAE (£) | RMSE (£) | R² |
|-------|---------|----------|----|
| Linear Regression | 124,426 | 167,603 | 0.188 |
| Ridge Regression | 124,404 | 167,613 | 0.187 |
| Random Forest | 40,176 | 46,227 | 0.938 |
| **Gradient Boosting** | **38,048** | **44,391** | **0.943** |

**Best model: Gradient Boosting** — R² = 0.943, MAE = £38,048

---

## Key Findings

- **Region** is the strongest predictor (63% feature importance), followed by **Property Type** (29%)
- London Detached averages **£1,097k** in 2025 — nearly 8× the cheapest segment (North East Flat at £141k)
- All regions peaked around **2022** then corrected slightly
- Tree-based models vastly outperform linear models due to non-linear regional price effects

---

## Output Charts

| Chart | Description |
|-------|-------------|
| `price_by_region.png` | Price trends for all 12 regions (2015–2025) |
| `price_by_property_type.png` | Price trends by Detached / Semi / Terraced / Flat |
| `heatmap_region_type.png` | Region × Property Type price heatmap (2025) |
| `model_comparison.png` | R² comparison across all 4 models |
| `actual_vs_predicted.png` | Scatter plot of predictions vs actuals |
| `feature_importance.png` | Random Forest feature importances |

---

## Project Structure

```
uk-house-price-prediction/
├── uk_house_prices_raw.csv       # Raw dataset
├── main.py                       # Full ML pipeline script
├── main_notebook.ipynb           # Notebook version of main.py
├── house_price_prediction.ipynb  # Extended notebook with error analysis
├── requirements.txt              # Python dependencies
└── outputs/                      # Generated charts
    ├── price_by_region.png
    ├── price_by_property_type.png
    ├── heatmap_region_type.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

---

## Setup & Usage

**1. Create and activate a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3a. Run the script**
```bash
python main.py
```

**3b. Or open the notebook**
```bash
jupyter notebook main_notebook.ipynb
```

---

## Predict a Price

```python
predict_price(year=2025, month=6, region="London", property_type="Detached", sales_volume=600)
# -> £974,793
```

**Available regions:** London, South East, South West, East of England, East Midlands, West Midlands, North West, North East, Yorkshire and the Humber, Scotland, Wales, Northern Ireland

**Available property types:** Detached, Semi-Detached, Terraced, Flat

---

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, jupyter
