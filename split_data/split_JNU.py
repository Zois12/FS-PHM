import numpy as np 
import pandas as pd
from pathlib import Path
import h5py
import re

CSV_dir = Path("../江南大学轴承数据集/JNU-Bearing-Dataset-main")
OUT_dir = Path("JNU_h5")
OUT_dir.mkdir(parents=True,exist_ok=True)
SAMPLING_RATE = 50000
WINDOW_SEC = 0.2
STRIDE_RATIO = 0.5
WINDOW = int(SAMPLING_RATE * WINDOW_SEC)
STRIDE = max(1, int(WINDOW * STRIDE_RATIO))

def make_window(signal,window,stride):
    n = len(signal)
    if n < window:
        return None
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:,None] + np.arange(window)[None, :]
    return signal[idx].astype(np.float32)

def parse_cond_label(filename):
    stem = Path(filename).stem

    m = re.match(r'^([a-zA-Z]+)(\d+)', stem)
    if not m:
        raise ValueError(f"文件名不符合规则: {filename}")
    letters, speed = m.group(1), int(m.group(2))

    cond = letters[0]   
    COND_TO_LABEL = {
    "i": 1, # 内圈故障
    "o": 2, # 外圈故障
    "t": 3, # 滚动体故障
    "n": 0, # 正常轴承
    }
    label = COND_TO_LABEL.get(cond,-1)
    outer_diameter = 52
    inter_diameter = 25
    roller_diameter = 7
    contact_angle = 0
    roller_num = 10
    if label == 1:
        roller_num = 11

    return {
        "label":label,
        "speed_rpm":speed,
        "outer_diameter":outer_diameter,
        "inter_diameter":inter_diameter,
        "roller_diameter":roller_diameter,
        "contact_angle":contact_angle,
        "roller_num":roller_num
    }

csv_files = sorted(CSV_dir.rglob("*.csv"))
print(f"共{len(csv_files)}个文件")

groups = {}
existing = {p.name for p in OUT_dir.glob("*.h5")}
print(f"已存在 {len(existing)} 个 h5，将跳过")

for csv_path in csv_files:
    cond = parse_cond_label(csv_path.name)
    key = f"{csv_path.stem.split('_')[0]}"   # 直接取 'ib600' / 'n600' 这种前缀
    groups.setdefault(key, {
        "info":  cond,
        "files": [],
    })["files"].append(csv_path)

print(f"共 {len(groups)} 个组: {sorted(groups.keys())}")

for key, g in groups.items():
    out_name = f"{key}.h5"
    out_path = OUT_dir / out_name

    if out_name in existing:
        print(f"[{key}] 已存在，跳过")
        continue

    info  = g["info"]
    label = info["label"]
    if label < 0:
        print(f"[{key}] 标签未定义，跳过")
        continue

    with h5py.File(out_path, "w") as f:
        ds_sig = f.create_dataset(
            "signal", shape=(0, WINDOW), maxshape=(None, WINDOW),
            dtype="f4", compression="gzip", compression_opts=4,
        )
        ds_lb = f.create_dataset(
            "label", shape=(0,), maxshape=(None,), dtype="i8",
        )

        offset = 0
        for csv_path in g["files"]:
            try:
                df = pd.read_csv(csv_path, header=None).dropna(axis=1, how="all")
                col = df.count().idxmax()
                arr = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(np.float32)
            except Exception as e:
                print(f"  读取失败: {csv_path.name}  {e}")
                continue

            win = make_window(arr, WINDOW, STRIDE)
            if win is None:
                print(f"  {csv_path.name} 长度不足，跳过")
                continue

            m = win.shape[0]
            ds_sig.resize(offset + m, axis=0)
            ds_lb.resize(offset + m, axis=0)
            ds_sig[offset:offset+m] = win
            ds_lb[offset:offset+m]  = np.full(m, label, dtype=np.int64)
            offset += m

            print(f"  {csv_path.name:20s} -> {m:>4d} 窗口")

        # 写 attrs
        f.attrs["condition"]       = key
        f.attrs["label"]           = label
        f.attrs["speed_rpm"]       = info["speed_rpm"]
        f.attrs["sampling_rate"]   = SAMPLING_RATE
        f.attrs["window"]          = WINDOW
        f.attrs["stride"]          = STRIDE
        f.attrs["outer_diameter"]  = info["outer_diameter"]
        f.attrs["inter_diameter"]  = info["inter_diameter"]
        f.attrs["roller_diameter"] = info["roller_diameter"]
        f.attrs["roller_num"]      = info["roller_num"]
        f.attrs["contact_angle"]   = info["contact_angle"]
        f.attrs["num_samples"]     = offset

    print(f"[{key}] 共 {offset} 窗口 -> {out_name}")

print("\n全部完成。")