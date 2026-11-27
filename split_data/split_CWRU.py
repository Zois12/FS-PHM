import argparse
import numpy as np
import re 
from pathlib import Path
from scipy.io import loadmat
import h5py
try:
    from .cwru_provenance import register_acquisition
except ImportError:
    from cwru_provenance import register_acquisition

# 数据路径 
mapping = {
    "Normal":    ("../CWRU/Normal Baseline Data",              12000),
    "12k_drive": ("../CWRU/12k Drive End Bearing Fault Data",  12000),
    "48k_drive": ("../CWRU/48k Drive End Bearing Fault Data",  48000),
    "12k_fan":   ("../CWRU/12k Fan End Bearing Fault Data",    12000),
}

WINDOW_SEC = 0.2 
STRIDE_RATIO = 0.5

sets = ["Normal","12k_fan","12k_drive","48k_drive"]
out_dir = Path("CWRU_h5")
 # 故障标签
 # 0：正常
 # 1：IR007  5:B007  9:OR007@6  13:OR007@3  17:OR007@12
 # 2：IR014  6:B014  10:OR014@6 14:OR014@3  18:OR014@12
 # 3: IR021  7:B021  11:OR021@6 15:OR021@3  19:OR021@12
 # 4: IR028  8:B028  12:OR028@6 16:OR028@3  20:OR028@12
RPM_MAP      = {"0": 1797, "1": 1772, "2": 1750, "3": 1730}

def group_by_device(data):
    """
    输入loadmat的结果 输出
    {
        "X098": {"DE": "X098_DE_time", "FE": "X098_FE_time", "BA": None, "RPM": "X098RPM"},
        "X099": {...},
      }
    规则:
      - 变量名形如 X###_DE_time / X###_FE_time / X###_BA_time / X###RPM
      - 以 X 后面的编号作为分组 key
    """
    devices = {}

    for key in data.keys():
        if key.startswith("__") or key == "ans":
            continue
        m = re.match(r'(X\d+)', key)
        if not m:
            continue
        dev = m.group(1)
        devices.setdefault(dev, {"DE": None, "FE": None, "BA": None, "RPM": None})

        if "DE_time" in key:
            devices[dev]["DE"] = key
        elif "FE_time" in key:
            devices[dev]["FE"] = key
        elif "BA_time" in key:
            devices[dev]["BA"] = key
        elif "RPM" in key:
            devices[dev]["RPM"] = key
    return devices

def rpm_from_filename(filename):
    hp = Path(filename).stem.split("_")[-1]
    return RPM_MAP.get(hp, 0)

def make_windows(signal, window, stride):
    """ 一维信号 -> (M, window)"""
    n = len(signal)
    if n < window:
        return None
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    return signal[idx]

def get_label(set_name,filename):
    if set_name == "Normal":
        return 0
    name = Path(filename).stem 
    m = re.match(r'(IR|B|OR)(\d+)',name)
    if not m:
        return -1
    fault_type = m.group(1)
    size = int(m.group(2))
    size_idx = {7: 0, 14: 1, 21: 2, 28: 3}[size]

    if fault_type == "IR":
        return 1 + size_idx
    if fault_type == "B":
        return 5 + size_idx

    pos_m = re.search(r'@(\d+)', name)
    pos = int(pos_m.group(1)) if pos_m else 6
    base = {6: 9, 3: 13, 12: 17}[pos]
    return base + size_idx

def build_set(set_name,folder,sampling_rate):
    folder = Path(folder)
    mat_files = sorted(folder.glob("*.mat"))

    window = int(sampling_rate * WINDOW_SEC)
    stride = max(1, int(window * STRIDE_RATIO))

    out_sub = out_dir / set_name 
    out_sub.mkdir(parents=True,exist_ok=True)

    print(f"\n[{set_name}] {len(mat_files)} 个文件，采样率{sampling_rate} Hz,"
          f"window={window},stride={stride}")

    total_files = 0
    total_windows = 0 
    seen_acquisitions = {}

    for mat_path in mat_files:
        data = loadmat(mat_path)
        devices = group_by_device(data)
        if not devices:
            print(f" 跳过{mat_path.name}(没找到设备变量)")
            continue

        rpm = rpm_from_filename(mat_path.name)
        label = get_label(set_name, mat_path.name)

        for dev_name, keys in devices.items():
            if keys["DE"] is None:
                continue

            de = np.asarray(data[keys["DE"]].squeeze().astype(np.float32))
            if keys["FE"] is not None:
                fe = np.asarray(data[keys["FE"]]).squeeze().astype(np.float32)
                n = min(len(de), len(fe))
                de, fe = de[:n], fe[:n]
            else:
                fe = np.zeros_like(de)

            fingerprint, duplicate = register_acquisition(
                seen_acquisitions, de, fe, label, sampling_rate, f"{mat_path.name}:{dev_name}")
            if duplicate is not None:
                print(f"跳过重复原始记录 {mat_path.name}:{dev_name}；已保留 {duplicate}")
                continue

            de_win = make_windows(de, window, stride)
            if de_win is None:
                print(f" [{mat_path.name} | {dev_name}]长度{len(de)} < window,跳过")
                continue 
            m = de_win.shape[0]

            fe_win = make_windows(fe, window, stride)

            rpm_arr = np.full(m,rpm, dtype=np.float32)
            label_arr = np.full(m, label, dtype=np.int64)
            h5_path = out_sub / f"{mat_path.stem}_{dev_name}.h5"
            
            with h5py.File(h5_path, "w") as f:
                f.attrs["set_name"]      = set_name
                f.attrs["sampling_rate"] = sampling_rate
                f.attrs["window"]        = window 
                f.attrs["stride"]        = stride
                f.attrs["window_sec"]    = WINDOW_SEC
                f.attrs["source"]        = mat_path.name
                f.attrs["source_signal_sha256"] = fingerprint
                f.attrs["device"]        = dev_name
                f.attrs["num_windows"]   = de_win.shape[0]
                f.create_dataset("DE", data=de_win,
                                 compression="gzip", compression_opts=4)
                f.create_dataset("FE", data=fe_win,
                                 compression="gzip", compression_opts=4)
                f.create_dataset("rpm",data=rpm_arr)
                f.create_dataset("label",data=label_arr)
                f.create_dataset("window_start", data=np.arange(m, dtype=np.int64) * stride)

            total_files += 1
            total_windows += m 
            print(f"    {mat_path.name:30s} | {dev_name} | "
                  f"len={len(de):>7d} -> {m:>5d} 窗口 | "
                  f"rpm={rpm} | label={label} -> {h5_path.name}")
    print(f"[{set_name}] 共 {total_files} 个 h5，{total_windows} 个窗口")

def main():
    global out_dir
    parser = argparse.ArgumentParser(description="Export CWRU raw acquisitions once, with provenance and overlapping windows")
    parser.add_argument("--output-root", default="CWRU_h5")
    parser.add_argument("--sets", nargs="+", choices=tuple(mapping), default=list(mapping))
    args = parser.parse_args()
    out_dir = Path(args.output_root)
    if out_dir.exists() and any(out_dir.rglob("*.h5")):
        raise ValueError("Use a new --output-root; existing H5 files will not be overwritten or mixed with stale duplicate files")
    for set_name in args.sets:
        folder, sr = mapping[set_name]
        build_set(set_name, folder, sr)


if __name__ == "__main__":
    main()
