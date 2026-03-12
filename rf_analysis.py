import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import json

DATA_DIR   = Path("/Users/chenjiayin/Desktop/claude/房價分析/data/台北市不動產和預售屋買賣資料/整合/分類篩選")
FILTER_DIR = Path("/Users/chenjiayin/Desktop/claude/房價分析/data/台北市不動產和預售屋買賣資料/整合/資料篩選")

# ── 輔助：移轉層次 → 樓層數字（與 geocode_yearly.py 同邏輯）─────────────
def zh_to_int(s):
    units = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
    if s in units: return units[s]
    if s == "十": return 10
    if "十" in s:
        parts = s.split("十")
        tens = units.get(parts[0], 1) if parts[0] else 1
        ones = units.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return None

def extract_floor(s):
    if pd.isna(s): return None
    s = str(s).strip()
    if s.startswith("地下"): return 0
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = re.match(r"(\d+)層", s)
    if m: return int(m.group(1))
    m = re.match(r"([一二三四五六七八九十]+)層", s)
    if m: return zh_to_int(m.group(1))
    return None

def strip_district(addr):
    """移除門牌中的「臺北市XX區」前綴，與 geocode_yearly.py 一致"""
    return re.sub(r"臺北市\S+區", "", str(addr)).strip()

# ── 建立指定年份的 (地址, 幾樓) → 季度 對照表 ────────────────────────────
def build_quarter_map(year):
    records = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        path = FILTER_DIR / year / f"{year}{q}不動產買賣.csv"
        if not path.exists():
            continue
        qdf = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                          usecols=["土地位置建物門牌", "移轉層次"])
        qdf["地址"]  = qdf["土地位置建物門牌"].apply(strip_district)
        qdf["幾樓"]  = qdf["移轉層次"].apply(extract_floor)
        qdf["季度"]  = q
        records.append(qdf[["地址", "幾樓", "季度"]])
        print(f"  [{year}{q}] 載入 {len(qdf)} 筆")

    if not records:
        return {}
    combined = pd.concat(records, ignore_index=True)
    combined = combined.drop_duplicates(subset=["地址", "幾樓"], keep="first")
    return combined.set_index(["地址", "幾樓"])["季度"].to_dict()

# ── 1. 載入分類篩選資料 ──────────────────────────────────────────────────
print("載入分類篩選資料...")
all_dfs = []
for csv_file in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file, encoding="utf-8")
    df["年份"] = csv_file.stem.split("_")[0]
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(df)}")

# ── 2. 為 114、115 年標記季度 ────────────────────────────────────────────
for year in ["114", "115"]:
    print(f"\n建立 {year} 年季度對照表...")
    quarter_map = build_quarter_map(year)
    print(f"  對照表共 {len(quarter_map)} 筆唯一 (地址, 幾樓)")

    mask_yr = df["年份"] == year
    df.loc[mask_yr, "季度"] = df.loc[mask_yr].apply(
        lambda r: quarter_map.get((r["地址"], r["幾樓"]), None), axis=1
    )
    matched  = df.loc[mask_yr, "季度"].notna().sum()
    total_yr = mask_yr.sum()
    if total_yr > 0:
        print(f"  {year} 年共 {total_yr} 筆，成功對應季度 {matched} 筆（{matched/total_yr:.1%}）")

# 未能對應的資料無法確認季度，一律歸入訓練集
df["季度"] = df["季度"].fillna("Q_unknown")

# ── 3. 特徵工程 ──────────────────────────────────────────────────────────
target_col   = "每坪價格"
feature_cols = ["幾樓", "屋齡", "是否有電梯", "坪數", "車位坪數",
                "緯度", "經度", "距離(公尺)", "交易標的", "類別", "年份"]

df_model = df[feature_cols + [target_col, "季度"]].copy()

le_trans = LabelEncoder()
le_cat   = LabelEncoder()
le_year  = LabelEncoder()

df_model["交易標的_enc"] = le_trans.fit_transform(df_model["交易標的"].fillna("Unknown"))
df_model["類別_enc"]    = le_cat.fit_transform(df_model["類別"].fillna("Unknown"))
df_model["年份_enc"]    = le_year.fit_transform(df_model["年份"].fillna("Unknown"))

numeric_feats = ["幾樓", "屋齡", "坪數", "車位坪數", "緯度", "經度", "距離(公尺)"]
for col in numeric_feats:
    df_model[col] = df_model[col].fillna(df_model[col].median())

df_model["是否有電梯"] = df_model["是否有電梯"].fillna(0).astype(int)

X_cols = ["幾樓", "屋齡", "是否有電梯", "坪數", "車位坪數",
          "緯度", "經度", "距離(公尺)", "交易標的_enc", "類別_enc", "年份_enc"]
X = df_model[X_cols]
y = df_model[target_col]

# ── 4. 移除極端值（依整體分布） ──────────────────────────────────────────
q_low  = y.quantile(0.01)
q_high = y.quantile(0.99)
mask   = (y >= q_low) & (y <= q_high)
X      = X[mask]
y      = y[mask]
meta   = df_model[["季度", "年份"]][mask].copy()
print(f"\n移除極端值後：{len(X)} 筆")

# ── 5. 時間序列切分：訓練 = ~Q3 2025，測試 = Q4 2025 + Q1 2026 ────────────
test_mask  = (
    ((meta["年份"] == "114") & (meta["季度"] == "Q4")) |
    ((meta["年份"] == "115") & (meta["季度"] == "Q1"))
)
train_mask = ~test_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
print(f"訓練集（2022–2025 Q3）       ：{len(X_train)} 筆")
print(f"測試集（2025 Q4 + 2026 Q1）  ：{len(X_test)} 筆")

# ── 6. 訓練 Random Forest ────────────────────────────────────────────────
print("\n訓練模型...")
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=5,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ── 7. 評估 ─────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(rf, X_train, y_train, cv=kf, scoring="r2")

print(f"\n=== Model Performance ===")
print(f"MAE:  {mae:,.0f} TWD/ping")
print(f"RMSE: {rmse:,.0f} TWD/ping")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"CV R² on train (5-fold): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# ── 8. 特徵重要性 ────────────────────────────────────────────────────────
feat_importance = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False)
print(f"\n=== Feature Importances ===")
print(feat_importance)

# ── 9. 統計 ─────────────────────────────────────────────────────────────
cat_stats  = df.groupby("類別")[target_col].agg(["count","mean","median","std"]).round(0)
year_stats = df.groupby("年份")[target_col].agg(["count","mean","median"]).round(0)

# ── 10. 輸出結果 ─────────────────────────────────────────────────────────
results = {
    "total_rows":  int(len(df)),
    "model_rows":  int(len(X)),
    "train_rows":  int(len(X_train)),
    "test_rows":   int(len(X_test)),
    "split_method": "time-based: train=2022-2025Q3, test=2025Q4+2026Q1",
    "mae":  float(mae),
    "rmse": float(rmse),
    "r2":   float(r2),
    "mape": float(mape),
    "cv_r2_mean": float(cv_r2.mean()),
    "cv_r2_std":  float(cv_r2.std()),
    "feature_importances": {k: float(v) for k, v in feat_importance.items()},
    "cat_stats": {
        row: {"count": int(cat_stats.loc[row,"count"]), "mean": float(cat_stats.loc[row,"mean"]),
              "median": float(cat_stats.loc[row,"median"]), "std": float(cat_stats.loc[row,"std"])}
        for row in cat_stats.index
    },
    "year_stats": {
        row: {"count": int(year_stats.loc[row,"count"]), "mean": float(year_stats.loc[row,"mean"]),
              "median": float(year_stats.loc[row,"median"])}
        for row in year_stats.index
    },
    "target_stats": {
        "min": float(y.min()), "max": float(y.max()),
        "mean": float(y.mean()), "median": float(y.median()), "std": float(y.std()),
    }
}

with open("/Users/chenjiayin/Desktop/claude/房價分析/rf_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nResults saved to rf_results.json")
