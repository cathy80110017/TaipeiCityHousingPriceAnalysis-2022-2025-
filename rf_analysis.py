import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import json

DATA_DIR = Path("/Users/chenjiayin/Desktop/claude/房價分析/data/台北市不動產和預售屋買賣資料/整合/分類篩選")

# 1. Load all CSV files
all_dfs = []
for csv_file in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file, encoding="utf-8")
    df["年份"] = csv_file.stem.split("_")[0]
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# 2. Basic stats
target_col = "每坪價格"
print(f"\nTarget column '{target_col}' stats:")
print(df[target_col].describe())

# 3. Feature engineering
feature_cols = ["幾樓", "屋齡", "是否有電梯", "坪數", "車位坪數", "緯度", "經度", "距離(公尺)", "交易標的", "類別", "年份"]

df_model = df[feature_cols + [target_col]].copy()

# Encode categorical columns
le_trans = LabelEncoder()
le_cat = LabelEncoder()
le_year = LabelEncoder()

df_model["交易標的_enc"] = le_trans.fit_transform(df_model["交易標的"].fillna("Unknown"))
df_model["類別_enc"] = le_cat.fit_transform(df_model["類別"].fillna("Unknown"))
df_model["年份_enc"] = le_year.fit_transform(df_model["年份"].fillna("Unknown"))

# Fill missing numeric values with median
numeric_feats = ["幾樓", "屋齡", "坪數", "車位坪數", "緯度", "經度", "距離(公尺)"]
for col in numeric_feats:
    median_val = df_model[col].median()
    df_model[col] = df_model[col].fillna(median_val)

# Binary feature
df_model["是否有電梯"] = df_model["是否有電梯"].fillna(0).astype(int)

# Final features
X_cols = ["幾樓", "屋齡", "是否有電梯", "坪數", "車位坪數", "緯度", "經度", "距離(公尺)", "交易標的_enc", "類別_enc", "年份_enc"]
X = df_model[X_cols]
y = df_model[target_col]

# Remove outliers (top/bottom 1%)
q_low = y.quantile(0.01)
q_high = y.quantile(0.99)
mask = (y >= q_low) & (y <= q_high)
X = X[mask]
y = y[mask]
print(f"\nAfter outlier removal: {len(X)} rows")

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# 5. Train Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(rf, X, y, cv=kf, scoring="r2")

print(f"\n=== Model Performance ===")
print(f"MAE:  {mae:,.0f} TWD/ping")
print(f"RMSE: {rmse:,.0f} TWD/ping")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"CV R² (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# 7. Feature importance
feat_importance = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
print(f"\n=== Feature Importances ===")
print(feat_importance)

# 8. Per-category stats
cat_stats = df.groupby("類別")[target_col].agg(["count", "mean", "median", "std"]).round(0)
print(f"\n=== Price by Category ===")
print(cat_stats)

# 9. Year-over-year trend
year_stats = df.groupby("年份")[target_col].agg(["count", "mean", "median"]).round(0)
print(f"\n=== Price by Year ===")
print(year_stats)

# Save results for report
results = {
    "total_rows": int(len(df)),
    "model_rows": int(len(X)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "mape": float(mape),
    "cv_r2_mean": float(cv_r2.mean()),
    "cv_r2_std": float(cv_r2.std()),
    "feature_importances": {k: float(v) for k, v in feat_importance.items()},
    "cat_stats": {row: {"count": int(cat_stats.loc[row, "count"]), "mean": float(cat_stats.loc[row, "mean"]), "median": float(cat_stats.loc[row, "median"]), "std": float(cat_stats.loc[row, "std"])} for row in cat_stats.index},
    "year_stats": {row: {"count": int(year_stats.loc[row, "count"]), "mean": float(year_stats.loc[row, "mean"]), "median": float(year_stats.loc[row, "median"])} for row in year_stats.index},
    "target_stats": {
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std()),
    }
}

with open("/Users/chenjiayin/Desktop/claude/房價分析/rf_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nResults saved to rf_results.json")
