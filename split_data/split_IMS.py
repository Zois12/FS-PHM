import numpy as np 
import pandas as pd 
import h5py 
import re 
from pathlib import Path 

ROOT = Path("../4. Bearings/IMS")
OUT_DIR = Path("ims_h5")
OUT_DIR.mkdir(parents=True,exist_ok=True)

SAMPLING_RATE = 20000
WINDOW_SEC = 0.1
STRIDE_RATIO = 0.5 
WINDOW = int(SAMPLING_RATE * WINDOW_SEC)
STRIDE = int(WINDOW * STRIDE_RATIO)

def read_ims_txt(txt_path):
    try:
        df = pd.read_csv(txt_path, sep=r"\s+", header=None, dtype=np.float32)
    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        print(f"  读取异常 {txt_path.name}: {e}")
        return None

    arr = df.values
    if arr.size == 0:
        return None
    return arr

def list_ims_files(bearing_dir):
    """列出目录下的所有文件，按文件名排序"""
    files = [p for p in bearing_dir.iterdir() if p.is_file()]
    return sorted(files, key=lambda p: p.name)

def split_columns(arr,test_name):
    """按test 拆成4个轴承"""
    if test_name == "1st_test":
        return{
            "Bearing1":arr[:,0:2],
            "Bearing2":arr[:,2:4],
            "Bearing3":arr[:,4:6],
            "Bearing4":arr[:,6:8]
        }
    else:
        return{
            "Bearing1":arr[:,0:1],
            "Bearing2":arr[:,1:2],
            "Bearing3":arr[:,2:3],
            "Bearing4":arr[:,3:4]
        }

def make_windows_2d(sig, window, stride):
    C, L = sig.shape
    if L < window:
        return None
    starts = np.arange(0, L - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]     # (m, window)
    return sig[:, idx].transpose(1, 0, 2)                  # (m, C, window)

dataset_dir = ["1st_test","2nd_test","4th_test"]

for dataset in dataset_dir:
    path = ROOT / dataset
    files = list_ims_files(path)
    n_files = len(files)

    print(f"\n=== {dataset}  文件数={n_files} ===")

    bearing_data = {"Bearing1":[],"Bearing2":[],"Bearing3":[],"Bearing4":[]}
    for i,f in enumerate(files):
        arr = read_ims_txt(f)
        if arr is None:
            print(f"  跳过空/坏文件: {f.name}")
            continue
        split = split_columns(arr,dataset)
        for bname,bdata in split.items():
            bearing_data[bname].append(bdata.T)

    for bname,lst in bearing_data.items():

        out_path = OUT_DIR /f"{dataset}_{bname}.h5"

        all_win = []
        all_rul_min = []
        all_rul_norm = []
        max_rul_min = (n_files - 1) * 10 

        for fi,sig in enumerate(lst):
            win = make_windows_2d(sig, WINDOW, STRIDE)
            if win is None:
                continue
            m = win.shape[0]

            # 该文件对应一个 RUL，段内所有窗口共享
            rul_min  = (n_files - 1 - fi) * 10
            rul_norm = rul_min / max_rul_min if max_rul_min > 0 else 0.0

            all_win.append(win)
            all_rul_min.append(np.full(m, rul_min,  dtype=np.int32))
            all_rul_norm.append(np.full(m, rul_norm, dtype=np.float32))
            
        X      = np.concatenate(all_win,     axis=0)
        y_min  = np.concatenate(all_rul_min, axis=0)
        y_norm = np.concatenate(all_rul_norm,axis=0)

        n_channels = X.shape[1]
        with h5py.File(out_path,"w") as f:
            f.create_dataset("signal", data=X, dtype="f4",
                             compression="gzip",compression_opts=4)
            f.create_dataset("rul_minutes", data=y_min,  dtype="i4")
            f.create_dataset("rul_norm",    data=y_norm, dtype="f4")
            f.attrs["dataset"]       = dataset
            f.attrs["bearing"]       = bname
            f.attrs["n_files"]       = n_files
            f.attrs["n_channels"]    = n_channels
            f.attrs["channels"]      = ["x", "y"][:n_channels]
            f.attrs["sampling_rate"] = SAMPLING_RATE
            f.attrs["window"]        = WINDOW
            f.attrs["stride"]        = STRIDE
            f.attrs["window_sec"]    = WINDOW_SEC
            f.attrs["interval_min"]  = 10
            f.attrs["num_samples"]   = X.shape[0]
            f.attrs["done"]          = True

        print(f"  [{out_path.name}] shape={X.shape}  "
              f"RUL(min) 范围 [{y_min.min()}, {y_min.max()}]")

print("\n全部完成。")