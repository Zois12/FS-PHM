import numpy as np 
import pandas as pd 
import h5py
from pathlib import Path
from scipy.io import loadmat

DATA_DIR = Path("../加拿大渥太华大学变工况振动数据/v43hmbwxpm-2/")
OUT_DIR = Path("UO_h5")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# 故障标签
label_mapping = {
    "healthy":0,
    "inner race":1,
    "outer race":2,
    "ball":3,
    "combination":4
}

SAMPLING_RATE = 200000
WINDOW_SEC = 0.2
STRIDE_RATIO = 0.5
WINDOW = int(WINDOW_SEC * SAMPLING_RATE)
STRIDE = int(STRIDE_RATIO * WINDOW)

def make_windows(signal, window, stride):
    n = len(signal)
    if n < window:
        return None
    starts = np.arange(0, n - window + 1, stride)
    idx = starts[:, None] + np.arange(window)[None, :]
    return signal[idx].astype(np.float32)

all_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
for d in all_dirs:
    # 处理单个数据集文件夹
    dir_name = d.name # 获取数集文件夹名字

    # 输出子目录
    out_sub = OUT_DIR / dir_name 
    out_sub.mkdir(parents=True,exist_ok=True)

    # 标签获取
    if "healthy" in d.name:
        # 正常轴承数据集
        label = label_mapping.get("healthy")
    elif "inner race" in d.name:
        # 内圈故障
        label = label_mapping.get("inner race")
    elif "outer race" in d.name:
        # 外圈故障
        label = label_mapping.get("outer race")
    elif "ball" in d.name:
        # 滚珠故障
        label = label_mapping.get("ball")
    else:
        # 混合故障
        label = label_mapping.get("combination")

    # 获取文件夹下所有mat文件
    mat_data_list = d.rglob("*.mat")
    for mat_data in mat_data_list:
        singel_data = loadmat(mat_data) # 读取单个mat文件        
        channel_1 = np.asarray(singel_data)
        channel_2 = np.asarray(singel_data)

