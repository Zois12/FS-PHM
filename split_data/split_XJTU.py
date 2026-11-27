import numpy as np
import pandas as pd 
import re
from pathlib import Path 
import h5py 


CSV_DIR = Path("../西交轴承数据/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets")
OUT_DIR = Path("XJTU-SY_h5_01")
OUT_DIR.mkdir(exist_ok=True)

SAMPLING_RATE = 25600
WINDOW_SEC = 0.1
STRIDE_RATIO = 0.5
WINDOW = int(SAMPLING_RATE * WINDOW_SEC) # 5120
STRIDE = max(1,int(WINDOW * STRIDE_RATIO)) # 2560
POINTS_PER_FILE = 32768



RPM_RID = {"35Hz12kN":(2100,"12kN"),"37.5Hz11kN":(2250,"11kN"),"40Hz10kN":(2400,"10kN")}

def make_windows(signal, window, stride):
    n = len(signal)
    if n < window:
        return None
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    return signal[idx].astype(np.float32)

def read_two_cols(csv_path):
    """返回(Horizontal,Vertical) 两个1D float32 数组""" 
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError(f"列数不足2:{csv_path.name}")

    h = pd.to_numeric(df.iloc[:,0], errors="coerce").dropna().values.astype(np.float32)
    v = pd.to_numeric(df.iloc[:,1], errors="coerce").dropna().values.astype(np.float32)
    return h,v 


bearing_dirs = sorted([
   p for p in CSV_DIR.rglob("Bearing*")
   if p.is_dir() and any(p.glob("*.csv"))
])
print(f"共{len(bearing_dirs)}个轴承文件夹")


for bdir in bearing_dirs:
    # 相对于根目录的路径，如 '35Hz12kN/Bearing1_1/1.csv'
    rel = bdir.relative_to(CSV_DIR)
    parts = rel.parts   # ('35Hz12kN', 'Bearing1_1')
    condition = parts[0]      # 工况：35Hz12kN
    bearing   = parts[1]      # 轴承：Bearing1_1
    # file_name = parts[2]      # 文件名：1.csv
    speed_rpm = RPM_RID.get(condition)[0]
    radial_force = RPM_RID.get(condition)[1]

    out_name = f"{condition}_{bearing}.h5"
    out_path = OUT_DIR / out_name
    if out_path.exists():
        print(f"{out_name} 已存在，跳过")
        continue

    csv_files = sorted(
        bdir.glob("*.csv"),
        key = lambda p: int(re.search(r'\d+',p.stem).group())
    )
    n_files = len(csv_files)
    print(f"\n==={condition} / {bearing} ({n_files}文件={n_files}分钟) ===")

    all_h, all_v, all_rul_min, all_rul_norm = [],[],[],[]

    for fi,csv_path in enumerate(csv_files):
        try:
            h,v = read_two_cols(csv_path)
        except Exception as e:
            print(f" 读取失败{csv_path.name}:{e}")
            continue

        n = min(len(h),len(v))
        h,v = h[:n],v[:n]

        wh = make_windows(h, WINDOW, STRIDE)
        wv = make_windows(v, WINDOW, STRIDE)

        if wh is None:
            print(f"    {csv_path.name} 长度不足{WINDOW} 跳过")
            continue
        m = wh.shape[0]

        rul_min = n_files - fi - 1
        rul_norm = rul_min / n_files if n_files > 0 else 0.0

        all_h.append(wh)
        all_v.append(wv)
        all_rul_min.append(np.full(m, rul_min, dtype=np.int32))
        all_rul_norm.append(np.full(m,rul_norm,dtype=np.float32))
    if not all_h:
        print(" 没有可用数据")
        continue
    X_h = np.concatenate(all_h, axis=0)
    X_v = np.concatenate(all_v, axis=0)
    y_min  = np.concatenate(all_rul_min,  axis=0)
    y_norm = np.concatenate(all_rul_norm, axis=0)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("Horizontal", data=X_h, dtype="f4",
                         compression="gzip", compression_opts=4)
        f.create_dataset("Vertical",   data=X_v, dtype="f4",
                         compression="gzip", compression_opts=4)
        f.create_dataset("rul_minutes", data=y_min,  dtype="i4")
        f.create_dataset("rul_norm",    data=y_norm, dtype="f4")

        f.attrs["condition"]       = condition
        f.attrs["bearing"]         = bearing
        f.attrs["n_files"]         = n_files
        f.attrs["total_minutes"]   = n_files
        f.attrs["sampling_rate"]   = SAMPLING_RATE
        f.attrs["window"]          = WINDOW
        f.attrs["stride"]          = STRIDE
        f.attrs["points_per_file"] = POINTS_PER_FILE
        f.attrs["num_samples"]     = X_h.shape[0]
        f.attrs["speed_rpm"]       = speed_rpm
        f.attrs["radial_force"]    = radial_force
    print(f"  -> {out_name}: {X_h.shape[0]} 窗口  "
          f"RUL(分钟) 范围=[{y_min.min()}, {y_min.max()}]")

print("\n全部完成。")