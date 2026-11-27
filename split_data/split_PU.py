import numpy as np
from scipy.io import loadmat
import rarfile
import io
from pathlib import Path
import h5py
rarfile.UNRAR_TOOL = r"D:\WinRAR\UnRAR.exe"

AD = {
    # Artificial damage
    "KA01":("OR","1","EDM",1),
    "KA03":("OR","2","electric engraver",3),
    "KA05":("OR","1","electric engraver",2),
    "KA06":("OR","2","electric engraver",3),
    "KA07":("OR","1","drilling",4),
    "KA08":("OR","2","drilling",5),
    "KA09":("OR","2","drilling",5),
    "KI01":("IR","1","EDM",6),
    "KI03":("IR","1","electric engraver",7),
    "KI05":("IR","1","electric engraver",7),
    "KI07":("IR","2","electric engraver",8),
    "KI08":("IR","2","electric engraver",8),
    # Health bearings
    
}
HB = {
    "K001","K002","K003","K004","K005","K006"
}

SAMPLING_RATE = 64000
WINDOW_SEC = 0.2
STRIDE_RATIO = 0.5

out_dir = Path("PU_h5")
out_dir.mkdir(exist_ok=True)

datalist = list(Path("../德国帕德博恩轴承数据集/").rglob("*.rar"))


def unpack_mat_struct(x):
    # structured ndarray
    if isinstance(x, np.ndarray) and x.dtype.names:
        if x.size == 1:
            item = x.ravel()[0]
            return {name: unpack_mat_struct(item[name]) for name in x.dtype.names}
        return [unpack_mat_struct(item) for item in x.ravel()]

    # numpy.void：structured array 的单个元素
    if isinstance(x, np.void):
        return {name: unpack_mat_struct(x[name]) for name in x.dtype.names}

    # 普通 ndarray
    if isinstance(x, np.ndarray):
        return x

    return x

def make_windows(signal, window, stride):
    n = len(signal)
    if n < window:
        return None
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    return signal[idx]

def h5_to_dict(obj):
    """把 h5py 的 Group/Dataset 递归转成 dict / ndarray"""
    if isinstance(obj, h5py.Group):
        return {k: h5_to_dict(obj[k]) for k in obj.keys()}
    elif isinstance(obj, h5py.Dataset):
        data = obj[()]
        # MATLAB v7.3 里字符串常是 bytes 数组，转成 str
        if isinstance(data, np.ndarray) and data.dtype.kind in ('S', 'U'):
            if data.size == 1:
                return data.item().decode('utf-8') if isinstance(data.item(), bytes) else str(data.item())
            return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in data.ravel()]
        return data
    else:
        return obj

def load_mat_any(data_bytes):
    """
    先试 scipy.io.loadmat（v7.2 及以下），失败则用 h5py（v7.3）。
    返回一个类似 loadmat 的 dict。
    """
    try:
        return loadmat(io.BytesIO(data_bytes))
    except (TypeError, NotImplementedError, ValueError):
        # v7.3 用 h5py 读
        with h5py.File(io.BytesIO(data_bytes), 'r') as f:
            return h5_to_dict(f)

def parse_condition(n):
    """
    n 形如 'K001/N15_M07_F10_K001_4.mat'
    返回 dict: {'speed_rpm': 1500, 'torque_nm': 0.7, 'force_n': 1000, 'measure_id': 4}
    """
    stem = Path(n).stem          # N15_M07_F10_K001_4
    parts = stem.split("_")      # ['N15', 'M07', 'F10', 'K001', '4']

    def code_to_val(code, digits):
        # N15 -> 1500, M07 -> 0.7, F10 -> 1000
        return int(code[1:]) * (10 ** (digits - len(code[1:])))

    speed   = int(parts[0][1:]) * 100      # N15 -> 1500, N09 -> 900
    torque  = int(parts[1][1:]) / 10       # M07 -> 0.7, M01 -> 0.1
    force   = int(parts[2][1:]) * 100      # F10 -> 1000, F04 -> 400
    measure = int(parts[4])                # 第几次测量

    return {
        "speed_rpm":  speed,
        "torque_nm":  torque,
        "force_n":    force,
        "measure_id": measure,
    }

# 数据全在Y中
def process(datalist):
    # 先扫一遍已有的h5文件
    existing = {p.name for p in out_dir.glob("*.h5")}
    print(f"已存在{len(existing)}个h5文件，将跳过")
    # 读取所有rar文件
    for rar_path in datalist:
        # 单个rar文件
        # 获取轴承名，确定故障类型
        bearings = Path(rar_path).stem
        # 根据轴承名获取对应的故障标签
        # 故障标签 
        # 0 正常
        # 1:OR 1 EDM                       6:IR 1 EDM 
        # 2:OR 1 electric engraver         7:IR 1 electric engraver
        # 3:OR 2 electric engraver         8:IR 2 electric engraver
        # 4:OR 1 drilling
        # 5:OR 2 drilling
        if bearings in AD:
            print("Artifical Damage Dataset")
            label = AD.get(bearings).__getitem__(3)
        elif bearings in HB:
            print("Health Bearings Dataset")
            label = 0 # 故障标签
        else:
            print(f"{bearings} is Accelerated Life time Dataset, not process")
            continue

        

        # 读取单个rar文件里面的每个mat
        with rarfile.RarFile(rar_path) as rf:
            for name in rf.namelist():
                # 读取单个mat文件
                n = name.replace("\\","/")
                if not n.lower().endswith(".mat"):
                    print("跳过非mat:",n)
                    continue

                base = n[:-4]                     # 去掉 .mat  -> "K001/N09_M07_F10_K001_1"
                base = base.rsplit("_", 1)[0]     # 去掉最后的 _数字 -> "K001/N09_M07_F10_K001"
                safe_base = base.replace("/", "_")# -> "K001_N09_M07_F10_K001"
                out_name = f"{safe_base}.h5"
                out_path = out_dir / out_name

                
                # 已存在跳过
                if out_name in existing:
                    print(f" 跳过（已处理）:{n}")
                    continue

                cond = parse_condition(n) 
                data = rf.read(n)
                try:
                    mat = load_mat_any(data)
                except Exception as e:
                    print(f"  读取失败: {n}  ->  {type(e).__name__}: {e}")
                    continue
                key = str(Path(n).stem)
                val = mat[key]

                d = unpack_mat_struct(val)
                # 所有数据都主要在Y中
                Y = d["Y"]
                # Y[0] force
                # Y[1] phase_current_1
                # Y[2] phase_current_2
                # Y[3] speed
                # Y[4] temp_2_bearing_module
                # Y[5] torque
                # Y[6] vibration_1

                # 振动数据
                vibration_data = Y[6]["Data"][0]

                # 电流数据
                phase_current_data_1 = Y[1]["Data"][0]
                phase_current_data_2 = Y[2]["Data"][0]

                # 窗口长度和步长
                WINDOW = int(SAMPLING_RATE * WINDOW_SEC)
                STRIDE = max(1,int(WINDOW * STRIDE_RATIO))

                # 滑动取数据
                vw = make_windows(vibration_data,  WINDOW, STRIDE)
                c1 = make_windows(phase_current_data_1, WINDOW, STRIDE)
                c2 = make_windows(phase_current_data_2, WINDOW, STRIDE)

                # 扩展label
                m = vw.shape[0]

                

                with h5py.File(out_path, "a") as f:
                    # 如果数据集不存在，创建可增长数据集
                    if "vibration" not in f:
                        f.create_dataset("vibration", shape=(0, WINDOW),
                                         maxshape=(None, WINDOW),
                                         dtype="f4", compression="gzip",
                                         compression_opts=4)
                        
                        f.attrs["bearing"]       = bearings
                        f.attrs["sampling_rate"] = SAMPLING_RATE
                        f.attrs["window"]        = WINDOW
                        f.attrs["stride"]        = STRIDE
                        # 物理信息
                        f.attrs["speed_rpm"]     = cond["speed_rpm"]
                        f.attrs["torque_nm"]     = cond["torque_nm"]
                        f.attrs["force_n"]       = cond["force_n"]

                    if "phase_current_1" not in f:
                        f.create_dataset("phase_current_1", shape=(0, WINDOW),
                                         maxshape=(None, WINDOW),
                                         dtype="f4", compression="gzip",
                                         compression_opts=4)
                    if "phase_current_2" not in f:
                        f.create_dataset("phase_current_2", shape=(0, WINDOW),
                                         maxshape=(None, WINDOW),
                                         dtype="f4", compression="gzip",
                                         compression_opts=4)
                    if "label" not in f:
                        f.create_dataset("label", shape=(0,),
                                         maxshape=(None,),
                                         dtype="i8")

                    offset = f["vibration"].shape[0]

                    f["vibration"].resize(offset + m, axis=0)
                    f["phase_current_1"].resize(offset + m, axis=0)
                    f["phase_current_2"].resize(offset + m, axis=0)
                    f["label"].resize(offset + m, axis=0)

                    f["vibration"][offset:offset+m]       = vw
                    f["phase_current_1"][offset:offset+m] = c1
                    f["phase_current_2"][offset:offset+m] = c2
                    f["label"][offset:offset+m]           = np.full(m, label, dtype=np.int64)

                    
                     

                print(f"  {n:45s} -> 追加 {m:>4d} 窗口到 {out_path.name}")
process(datalist)
