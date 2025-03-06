from stream_tools.model import SAHIDetector
import cv2
import numpy as np
import pandas as pd
import torch
from time import perf_counter_ns
from pathlib import Path
import os
from tqdm import tqdm

info = {
    'model': [],
    'inf_size': [],
    'inf_time_ms': [],
    'img_w': [],
    'img_h': [],
    'img_size_MB': [],
    'memory_used_MB': []
}
device = torch.device('cuda:0')
models = ['yolov8n.pt', 'yolov8m.pt', 'yolov8x.pt']
inf_sizes = [[640, 640], [1280, 1280]]
src = Path('/home/alex/nkbtech/hdd4/Datasets/rentgen/png/china_may_full_png/')
img_paths = list(src.iterdir())[:25]
imgs = [cv2.imread(str(img_p)) for img_p in img_paths]
for m in tqdm(models):
    for s in tqdm(inf_sizes, desc=m):
        cfg = {
            'model_path': m,
            'device': 'cuda:0',
            'inference_imgsz': s,
            'crop_size': s,
            'crop_overlap': [240, 240],
            'time_logging_period': 10,
            'inference_conf':0.25,
            'verbose':False,
        }

        detector = SAHIDetector(cfg)
        detector.initialize()
        times = []
        for i in range(25):
            start = perf_counter_ns()
            preds = detector([imgs[i]])
            stop = perf_counter_ns()
            h, w, _ = imgs[i].shape
            # times.append(stop-start)
            free, total = torch.cuda.mem_get_info(device)
            used_memory_MB = (total - free) / 1024 ** 2
            img_size_MB = os.path.getsize(img_paths[i]) / 1024 / 1024
            info['model'].append(m)
            info['inf_time_ms'].append((stop-start)/1e6)
            info['img_h'].append(h)
            info['img_w'].append(w)
            info['img_size_MB'].append(img_size_MB)
            info['memory_used_MB'].append(used_memory_MB)
            info['inf_size'].append(s[0])

info = pd.DataFrame(info)
info.to_csv(f'/home/alex/nkbtech/projects/stream_tools/notebooks/benchmark_{src.stem}.csv')