import sys
import pandas as pd
import os
import re
from geopy.geocoders import ArcGIS
from geopy.extra.rate_limiter import RateLimiter

# ── 參數：接受年份 ─────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("用法：python geocode_yearly.py <年份>")
    print("例如：python geocode_yearly.py 111")
    sys.exit(1)

YEAR = sys.argv[1]
ROOT = "/Users/chenjiayin/Desktop/claude/房價分析/data/台北市不動產和預售屋買賣資料"
INPUT_PATH  = os.path.join(ROOT, "整合", "資料篩選", YEAR, f"{YEAR}.csv")
OUTPUT_PATH = os.path.join(ROOT, "整合", "新增座標", f"{YEAR}座標.csv")
CACHE_PATH  = os.path.join(ROOT, "整合", "新增座標", f"geocode_cache_{YEAR}.csv")

# ── Geopy 設定 ────────────────────────────────────────────────────
geolocator = ArcGIS(timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5, error_wait_seconds=5)

# ── 輔助函式 ──────────────────────────────────────────────────────

def zh_to_int(s):
    """中文數字（支援 1–99）→ int，失敗回傳 None"""
    units = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
    if s in units:
        return units[s]
    if s == "十":
        return 10
    if "十" in s:
        parts = s.split("十")
        tens = units.get(parts[0], 1) if parts[0] else 1
        ones = units.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return None

def extract_floor(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    # 地下室
    if s.startswith("地下"):
        return 0
    # 阿拉伯數字（含全形）
    s_half = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    m = re.match(r"(\d+)層", s_half)
    if m:
        return int(m.group(1))
    # 中文數字：取「層」之前的部分
    m = re.match(r"([一二三四五六七八九十]+)層", s)
    if m:
        return zh_to_int(m.group(1))
    return None

def calc_age(completion, transaction):
    try:
        c = int(str(completion).split(".")[0])
        t = int(str(transaction).split(".")[0])
        c_year = c // 10000
        t_year = t // 10000
        if c_year <= 0:
            return None
        return t_year - c_year
    except Exception:
        return None

def sqm_to_ping(sqm):
    try:
        v = float(sqm)
        return round(v / 3.30579, 2) if v > 0 else 0
    except Exception:
        return None

def price_per_ping(total_price, ping):
    try:
        return round(float(total_price) / ping) if ping and ping > 0 else None
    except Exception:
        return None

def net_price_per_ping(total_price, area_sqm, parking_price, parking_sqm):
    """每坪價格，扣除車位價格與面積"""
    try:
        net_price = float(total_price) - (float(parking_price) if not pd.isna(parking_price) else 0)
        net_area  = float(area_sqm)    - (float(parking_sqm)   if not pd.isna(parking_sqm)   else 0)
        net_ping  = net_area / 3.30579
        return round(net_price / net_ping) if net_ping > 0 else None
    except Exception:
        return None

# ── 載入快取 ──────────────────────────────────────────────────────
if os.path.exists(CACHE_PATH):
    cache_df = pd.read_csv(CACHE_PATH, encoding="utf-8-sig")
    key_col = "key" if "key" in cache_df.columns else "address"
    cache = dict(zip(cache_df[key_col], zip(cache_df["lat"], cache_df["lng"])))
    print(f"[{YEAR}] 載入快取：{len(cache)} 筆")
else:
    cache = {}

def build_query(district, address):
    """補回區名、轉半形、移除樓層"""
    s = str(address)
    # 全形 → 半形
    s = s.translate(str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ－",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
    ))
    # 移除樓層（號之後的部分）
    s = re.sub(r"(\d+號).*", r"\1", s).strip()
    return f"台北市{district}{s}"

def get_coords(district, address):
    key = f"{district}|{address}"
    if key in cache:
        return cache[key]
    full_addr = build_query(district, address)
    try:
        loc = geocode(full_addr)
        result = (round(loc.latitude, 6), round(loc.longitude, 6)) if loc else (None, None)
    except Exception as e:
        print(f"  geocode 錯誤：{full_addr} → {e}")
        result = (None, None)
    cache[key] = result
    return result

def save_cache():
    records = [{"key": k, "lat": v[0], "lng": v[1]} for k, v in cache.items()]
    pd.DataFrame(records).to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")

# ── 讀取資料 ──────────────────────────────────────────────────────
path = INPUT_PATH
data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
print(f"[{YEAR}] 讀入 {len(data)} 筆")

# ── 欄位計算 ──────────────────────────────────────────────────────
data["幾樓"] = data["移轉層次"].apply(extract_floor)
data["屋齡"] = data.apply(
    lambda r: r["屋齡"] if "屋齡" in r and pd.notna(r["屋齡"])
    else calc_age(r["建築完成年月"], r["交易年月日"]), axis=1
)
data["坪數"] = data["建物移轉總面積平方公尺"].apply(sqm_to_ping)
data["車位坪數"] = data["車位移轉總面積平方公尺"].apply(sqm_to_ping)
data["每坪價格"] = data.apply(
    lambda r: net_price_per_ping(r["總價元"], r["建物移轉總面積平方公尺"],
                                  r["車位總價元"], r["車位移轉總面積平方公尺"]), axis=1
)
data["車位每坪價格"] = data.apply(
    lambda r: price_per_ping(r["車位總價元"], r["車位坪數"])
    if r.get("車位坪數", 0) and r["車位坪數"] > 0 else None, axis=1
)
data["是否有電梯"] = data["電梯"].apply(lambda x: "有" if str(x).strip() == "有" else "無")

# ── Geocoding ─────────────────────────────────────────────────────
addr_pairs = data[["鄉鎮市區", "土地位置建物門牌"]].dropna().drop_duplicates().values.tolist()
uncached = [(d, a) for d, a in addr_pairs if f"{d}|{a}" not in cache]
print(f"[{YEAR}] 需 geocode：{len(uncached)} 筆（共 {len(addr_pairs)} 筆唯一地址）")

for i, (district, addr) in enumerate(uncached, 1):
    get_coords(district, addr)
    if i % 100 == 0:
        print(f"[{YEAR}] 進度：{i}/{len(uncached)}")
        save_cache()

save_cache()
print(f"[{YEAR}] 快取已儲存：{len(cache)} 筆")

# ── 套用座標並輸出 ────────────────────────────────────────────────
data["緯度"] = data.apply(
    lambda r: get_coords(r["鄉鎮市區"], r["土地位置建物門牌"])[0]
    if pd.notna(r["土地位置建物門牌"]) else None, axis=1
)
data["經度"] = data.apply(
    lambda r: get_coords(r["鄉鎮市區"], r["土地位置建物門牌"])[1]
    if pd.notna(r["土地位置建物門牌"]) else None, axis=1
)

output_cols = ["交易標的", "土地位置建物門牌", "幾樓", "屋齡", "是否有電梯",
               "坪數", "每坪價格", "車位坪數", "車位每坪價格", "緯度", "經度"]
output = data[output_cols].rename(columns={"土地位置建物門牌": "地址"})
output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

total = len(output)
geocoded = output["緯度"].notna().sum()
print(f"\n[{YEAR}] ✅ 完成！{total} 筆，geocode 成功 {geocoded} 筆（{geocoded/total:.1%}）")
print(f"[{YEAR}] 輸出：{OUTPUT_PATH}")
