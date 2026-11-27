import numpy as np
import pandas as pd 
import re 
import h5py 
from pathlib import Path

SAMPLING_RATE = 25600
WINDOW = 2560
INTERVAL_SEC = 10

CSV_DIR = Path("../PHM 2012/phm-ieee-2012-data-challenge-dataset")
OUT_DIR = Path("PHM2012_h5")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# 运行状态 rpm / 径向力
condition = {"Bearing1":(1800,4000),"Bearing2":(1650,4200),"Bearing3":(1500,5000)}

SUBSETS = ["Learning_set","Full_Test_Set"]

def parse_acc_file(csv_path):
    """读取csv 返回震动数据 第五列=水平加速度 第六列 = 垂直加速度"""
    try:
        df = pd.read_csv(csv_path,header = None, sep = None, engine="python")
    except Exception as e:
        print(f" 读取失败{csv_path.name}:{e}")
        return None 

    if df.shape[1] < 6:
        print(f" {csv_path.name} 列数不足6 实际{df.shape[1]} 跳过")
        return None 

    h = pd.to_numeric(df.iloc[:,4],errors="coerce").values.astype(np.float32)
    v = pd.to_numeric(df.iloc[:,5],errors="coerce").values.astype(np.float32)

    n = min(len(h),len(v))
    h, v = h[:n], v[:n]

    if n < WINDOW:
        pad = WINDOW - n 
        h = np.pad(h, (0,pad))
        v = np.pad(v, (0,pad))
    else: 
        h,v = h[:WINDOW],v[:WINDOW]

    return np.stack([h, v],axis=0)

def list_acc_files(bearing_dir):
    """列出轴承目录下的 acc_*.csv，并按文件名里的数字排序"""
    files = list(bearing_dir.glob("acc_*.csv"))
    def key(p):
        m = re.search(r'\d+', p.stem)
        return int(m.group()) if m else 0
    return sorted(files, key=key)

# ---------------- 处理单个轴承 ----------------
def process_bearing(bearing_dir, subset):
    bearing = bearing_dir.name
    acc_files = list_acc_files(bearing_dir)
    if not acc_files:
        print(f"  [{subset}/{bearing}] 没有 acc 文件，跳过")
        return

    n_files = len(acc_files)
    out_name = f"{subset}_{bearing}.h5"
    out_path = OUT_DIR / out_name

    if out_path.exists():
        print(f"  [{out_name}] 已存在，跳过")
        return

    signals  = []
    rul_sec  = []
    rul_norm = []

    for i, acc_path in enumerate(acc_files):
        sig = parse_acc_file(acc_path)
        if sig is None:
            continue
        signals.append(sig)

        # 最后一个文件的 RUL = 0；每个文件间隔 10 秒
        r_sec = (n_files - 1 - i) * INTERVAL_SEC
        r_norm = r_sec / ((n_files - 1) * INTERVAL_SEC) if n_files > 1 else 0.0

        rul_sec.append(r_sec)
        rul_norm.append(r_norm)

    if not signals:
        print(f"  [{subset}/{bearing}] 无有效数据，跳过")
        return

    X = np.stack(signals, axis=0)                      # (N, 2, WINDOW)
    y_sec  = np.array(rul_sec,  dtype=np.int32)
    y_norm = np.array(rul_norm, dtype=np.float32)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("signal",      data=X,      dtype="f4",
                         compression="gzip", compression_opts=4)
        f.create_dataset("rul_seconds", data=y_sec,  dtype="i4")
        f.create_dataset("rul_norm",    data=y_norm, dtype="f4")

        f.attrs["subset"]        = subset
        f.attrs["bearing"]       = bearing
        f.attrs["n_files"]       = n_files
        f.attrs["sampling_rate"] = SAMPLING_RATE
        f.attrs["window"]        = WINDOW
        f.attrs["channels"]      = ["horizontal", "vertical"]
        f.attrs["num_samples"]   = X.shape[0]
        f.attrs["done"]          = True

    print(f"  [{out_name}] {X.shape[0]} 样本  "
          f"RUL 范围 [{y_sec.min()}, {y_sec.max()}] 秒")


# ---------------- 主流程 ----------------
if __name__ == "__main__":
    for subset in SUBSETS:
        subset_dir = CSV_DIR / subset
        if not subset_dir.exists():
            print(f"子集不存在，跳过: {subset_dir}")
            continue

        bearing_dirs = sorted([
            p for p in subset_dir.iterdir()
            if p.is_dir() and any(p.glob("acc_*.csv"))
        ])
        print(f"\n=== {subset}: {len(bearing_dirs)} 个轴承 ===")

        for bdir in bearing_dirs:
            process_bearing(bdir, subset)

    print("\n全部完成。")