import sys
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two coordinates."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# Get year from argument or prompt
if len(sys.argv) > 1:
    year = sys.argv[1]
else:
    year = input("請輸入年份（例如 111、112、113、114）：").strip()

input_path = f"data/台北市不動產和預售屋買賣資料/整合/新增座標/{year}座標.csv"
output_path = f"data/台北市不動產和預售屋買賣資料/整合/含捷運站距離/{year}座標_最近捷運站.csv"

print(f"讀取：{input_path}")
addresses = pd.read_csv(input_path)
mrt = pd.read_csv("data/臺北捷運車站出入口座標.csv")

addresses = addresses.dropna(subset=["緯度", "經度"])
mrt = mrt.dropna(subset=["緯度", "經度"])

mrt_lats = mrt["緯度"].values
mrt_lons = mrt["經度"].values
mrt_names = mrt["出入口名稱"].values

nearest_stations = []
nearest_distances = []

for _, row in addresses.iterrows():
    lat, lon = row["緯度"], row["經度"]
    distances = haversine(lat, lon, mrt_lats, mrt_lons)
    idx = np.argmin(distances)
    nearest_stations.append(mrt_names[idx])
    nearest_distances.append(round(distances[idx]))

addresses["最近捷運站出口"] = nearest_stations
addresses["距離(公尺)"] = nearest_distances

print(addresses[["地址", "最近捷運站出口", "距離(公尺)"]].head(10).to_string(index=False))
addresses.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n共處理 {len(addresses)} 筆資料，結果已儲存至 {output_path}")
