"""
UK House Price Fluctuation Prediction
Dataset: 2015-2025, 12 regions, 4 property types
Target: AveragePrice_GBP
"""

import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── 1. Load & inspect ──────────────────────────────────────────────────────────
df = pd.read_csv("uk_house_prices_raw.csv")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Year range   : {df['Year'].min()} – {df['Year'].max()}")
print(f"Regions      : {df['Region'].nunique()}")
print(f"PropertyTypes: {df['PropertyType'].nunique()}")
print("=" * 60)

# ── 2. Feature engineering ────────────────────────────────────────────────────
# Encode categorical columns
le_region = LabelEncoder()
le_ptype  = LabelEncoder()
df["Region_enc"]  = le_region.fit_transform(df["Region"])
df["PType_enc"]   = le_ptype.fit_transform(df["PropertyType"])

# Time features: months since Jan 2015 (captures long-run trend)
df["MonthsSince2015"] = (df["Year"] - 2015) * 12 + df["Month"]

# Seasonal: quarter
df["Quarter"] = ((df["Month"] - 1) // 3) + 1

# Log-transform the target (reduces skew, helps linear models)
df["LogPrice"] = np.log1p(df["AveragePrice_GBP"])

FEATURES = [
    "Year", "Month", "Quarter", "MonthsSince2015",
    "Region_enc", "PType_enc",
    "SalesVolume",
]
TARGET = "AveragePrice_GBP"

X = df[FEATURES]
y = df[TARGET]

# ── 3. Train / test split (temporal: last 18 months as test) ──────────────────
cutoff = df["MonthsSince2015"].max() - 18
train_mask = df["MonthsSince2015"] <= cutoff
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

# ── 4. Train models ───────────────────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0)),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=4,
        n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_leaf=4, subsample=0.8, random_state=42
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model, "preds": preds}
    print(f"\n{name}")
    print(f"  MAE  : £{mae:,.0f}")
    print(f"  RMSE : £{rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

# Pick best model by R²
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name}  (R² = {results[best_name]['R2']:.4f})")

# ── 5. Visualisations ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")

# 5a. Average price trends by region
fig, ax = plt.subplots(figsize=(14, 6))
region_trend = (
    df.groupby(["Year", "Region"])["AveragePrice_GBP"].mean().reset_index()
)
for region, grp in region_trend.groupby("Region"):
    ax.plot(grp["Year"], grp["AveragePrice_GBP"] / 1000, marker="o", markersize=3, label=region)
ax.set_title("Average House Price by Region (2015–2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Average Price (£ thousands)")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/price_by_region.png", dpi=150)
plt.close()

# 5b. Property type comparison
fig, ax = plt.subplots(figsize=(10, 5))
ptype_trend = (
    df.groupby(["Year", "PropertyType"])["AveragePrice_GBP"].mean().reset_index()
)
for ptype, grp in ptype_trend.groupby("PropertyType"):
    ax.plot(grp["Year"], grp["AveragePrice_GBP"] / 1000, marker="s", markersize=4, label=ptype)
ax.set_title("Average Price by Property Type (2015–2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Average Price (£ thousands)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/price_by_property_type.png", dpi=150)
plt.close()

# 5c. Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 4))
names  = list(results.keys())
r2vals = [results[n]["R2"] for n in names]
colors = ["steelblue" if n != best_name else "darkorange" for n in names]
bars = ax.bar(names, r2vals, color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel("R² Score")
ax.set_title("Model Comparison (R² on Test Set)")
for bar, val in zip(bars, r2vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()

# 5d. Actual vs Predicted (best model)
best_preds = results[best_name]["preds"]
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test / 1000, best_preds / 1000, alpha=0.4, edgecolors="none", s=20)
mn = min(y_test.min(), best_preds.min()) / 1000
mx = max(y_test.max(), best_preds.max()) / 1000
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
ax.set_xlabel("Actual Price (£ thousands)")
ax.set_ylabel("Predicted Price (£ thousands)")
ax.set_title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
plt.close()

# 5e. Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
feat_imp.plot(kind="barh", ax=ax, color="teal")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

print("\nPlots saved to outputs/")

# ── 6. Price heatmap: region × property type (most recent year) ───────────────
latest_year = df["Year"].max()
pivot = (
    df[df["Year"] == latest_year]
    .groupby(["Region", "PropertyType"])["AveragePrice_GBP"]
    .mean()
    .unstack("PropertyType")
    / 1000
)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title(f"Average Price (£k) by Region & Type — {latest_year}")
plt.tight_layout()
plt.savefig("outputs/heatmap_region_type.png", dpi=150)
plt.close()

# ── 7. Prediction function ────────────────────────────────────────────────────
def predict_price(year, month, region, property_type, sales_volume=500):
    """Predict average house price for given inputs using the best model."""
    region_enc = le_region.transform([region])[0]
    ptype_enc  = le_ptype.transform([property_type])[0]
    months_since = (year - 2015) * 12 + month
    quarter = ((month - 1) // 3) + 1
    row = pd.DataFrame([{
        "Year": year, "Month": month, "Quarter": quarter,
        "MonthsSince2015": months_since,
        "Region_enc": region_enc, "PType_enc": ptype_enc,
        "SalesVolume": sales_volume,
    }])
    price = best_model.predict(row)[0]
    return price


# Demo predictions
print("\n-- Predictions using best model --")
demo_cases = [
    (2025, 6,  "London",        "Detached",     600),
    (2025, 6,  "London",        "Flat",          800),
    (2025, 6,  "North East",    "Terraced",      400),
    (2025, 6,  "South East",    "Semi-Detached", 500),
    (2025, 6,  "Scotland",      "Flat",          350),
    (2025, 12, "London",        "Detached",      550),
]
for yr, mo, reg, ptype, vol in demo_cases:
    pred = predict_price(yr, mo, reg, ptype, vol)
    print(f"  {yr}-{mo:02d} | {reg:<25} | {ptype:<14} -> GBP{pred:>10,.0f}")

print("\nDone.")
