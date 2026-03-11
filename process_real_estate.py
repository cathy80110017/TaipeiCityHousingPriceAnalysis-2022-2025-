import pandas as pd
import os
import glob

ROOT = "/Users/chenjiayin/Desktop/claude/房價分析/data/台北市不動產和預售屋買賣資料"
BASE_DIR = os.path.join(ROOT, "原始資料")
OUTPUT_BASE = os.path.join(ROOT, "整合", "資料篩選")

YEARS = ["111", "112", "113", "114", "115"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

def find_q_folder(year, q):
    pattern = os.path.join(BASE_DIR, year, f"{q}*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def agg_build(build_df):
    """
    Build 檔一筆 編號 可能有多行（多棟建物）。
    彙總為一行：
    - 主要用途_build：取第一個非空值
    - 持分交易：移轉情形中只要有任一不是「全筆移轉」就標記 True
    """
    def first_valid(s):
        vals = s.dropna().astype(str).str.strip()
        vals = vals[vals != "nan"]
        return vals.iloc[0] if len(vals) > 0 else None

    def is_partial(s):
        return not all(v == "全筆移轉" for v in s.dropna().astype(str).str.strip() if v != "nan")

    agg = build_df.groupby("編號", as_index=False).agg(
        主要用途_build=("主要用途", first_valid),
        屋齡=("屋齡", first_valid),
        持分交易=("移轉情形", is_partial),
    )
    return agg

def process_quarter(year, q, folder):
    main_file = os.path.join(folder, "A_lvr_land_A.csv")
    build_file = os.path.join(folder, "A_lvr_land_A_build.csv")

    if not os.path.exists(main_file):
        print(f"  [{year}{q}] 找不到主檔，跳過")
        return None

    df = pd.read_csv(main_file, skiprows=[1], encoding="utf-8-sig", low_memory=False)
    print(f"  [{year}{q}] 讀入主檔：{len(df)} 筆")

    # 合併 build 檔（先彙總為每編號一行）
    if os.path.exists(build_file):
        build_raw = pd.read_csv(build_file, skiprows=[1], encoding="utf-8-sig", low_memory=False)
        build_agg = agg_build(build_raw)
        df = df.merge(build_agg, on="編號", how="left")
    else:
        df["主要用途_build"] = None
        df["持分交易"] = False

    assert df["編號"].is_unique, "merge 後 編號 不唯一！請檢查資料"

    # ── 1. 資料過濾 ──────────────────────────────────────────────
    before = len(df)

    # 特殊關係交易
    df = df[~df["備註"].astype(str).str.contains("親友|員工|共有人或其他特殊關係間之交易", na=False)]

    # 公共設施保留地
    df = df[~df["備註"].astype(str).str.contains("包含公共設施保留地用地", na=False)]

    # 特定交易標的：車位、土地
    df = df[~df["交易標的"].isin(["車位", "土地"])]

    # 非住宅用途
    def is_non_residential(row):
        urban = str(row.get("都市土地使用分區", "")).strip()
        main_use = str(row.get("主要用途", "")).strip()
        main_use_build = str(row.get("主要用途_build", "")).strip()
        if urban and urban not in ["nan", ""]:
            return urban != "住"
        else:
            use_ok = (main_use == "住家用") or (main_use_build == "住家用")
            return not use_ok

    df = df[~df.apply(is_non_residential, axis=1)]

    # 工業用
    df = df[df["主要用途_build"].astype(str).str.strip() != "工業用"]

    # 主要用途非住家用
    df = df[df["主要用途"] == "住家用"]

    print(f"  [{year}{q}] 過濾後：{len(df)} 筆（剔除 {before - len(df)} 筆）")

    # ── 2. 欄位調整 ──────────────────────────────────────────────
    drop_cols = ["地號", "主要建材", "車位所在樓層"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if "土地位置建物門牌" in df.columns:
        df["土地位置建物門牌"] = (
            df["土地位置建物門牌"]
            .astype(str)
            .str.replace(r"臺北市\S+區", "", regex=True)
            .str.strip()
        )

    # ── 3. 持分交易檢查 ──────────────────────────────────────────
    not_full = df["持分交易"].fillna(False)
    ratio = not_full.sum() / len(df) if len(df) > 0 else 0
    print(f"  [{year}{q}] 非完整持分比例：{ratio:.1%}（{not_full.sum()} 筆）")

    if ratio < 0.15:
        df = df[~not_full]
        print(f"  [{year}{q}] 比例 < 15%，已刪除持分交易")
    else:
        print(f"  [{year}{q}] ⚠️ 比例 >= 15%（{ratio:.1%}），保留持分交易，請注意！")

    # 移除暫時合併欄位
    df = df.drop(columns=[c for c in ["主要用途_build", "持分交易"] if c in df.columns])
    # 將 屋齡 轉為 float（first_valid 回傳 str）
    if "屋齡" in df.columns:
        df["屋齡"] = pd.to_numeric(df["屋齡"], errors="coerce")

    return df

# ── 主流程 ────────────────────────────────────────────────────────
for year in YEARS:
    for q in QUARTERS:
        folder = find_q_folder(year, q)
        if not folder:
            print(f"[{year}{q}] 資料夾不存在，跳過")
            continue

        print(f"\n處理 {year} {q}...")
        df = process_quarter(year, q, folder)
        if df is None or len(df) == 0:
            print(f"  [{year}{q}] 無有效資料")
            continue

        out_dir = os.path.join(OUTPUT_BASE, year)
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{year}{q}不動產買賣.csv"
        out_path = os.path.join(out_dir, out_name)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [{year}{q}] 已輸出：{out_path}（{len(df)} 筆）")

print("\n✅ 全部完成！")
