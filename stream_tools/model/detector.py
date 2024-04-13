import logging
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, cfg):
        # TODO add zones
        self.model = YOLO(model=cfg["model_path"], task="detect")
        self.device = torch.device(cfg['device'])
        self.cfg = cfg
    
    def initialize(self):
        # Dummy inference for model warmup
        for _ in range(100):
            dummy_imgs = [
                np.random.randint(
                    low=0,
                    high=255,
                    size=(self.cfg["orig_img_h"], self.cfg["orig_img_w"], 3),
                    dtype=np.uint8,
                )
                for _ in range(self.cfg["inference_bs"])
            ]
            self.model(
                source=dummy_imgs,
                device=self.device,
                imgsz=self.cfg["inference_imgsz"],
                conf=self.cfg["inference_conf"],
                stream=False,
                verbose=False,
                half=True,
            )
        self.time_logging_period = self.cfg["time_logging_period"]
        self.n_calls = -1

    @property
    def names(self):
        return self.model.names
    
    def __call__(self, imgs: list) -> Any:
        self.n_calls += 1
        return self.inference(imgs)

    def inference(self, imgs: list):
        start_time_ns = perf_counter_ns()
        correct_frame_idx = []
        for i in range(len(imgs)):
            if imgs[i] is None:
                continue
            # h, w, _ = imgs[i].shape
            correct_frame_idx.append(i)
            # if (h, w) != (
            #     self.cfg["orig_img_h"],
            #     self.cfg["orig_img_w"],
            # ):
            #     imgs[i] = cv2.resize(
            #         imgs[i],
            #         (self.cfg["orig_img_w"], self.cfg["orig_img_h"]),
            #         # Default YOLO interpolation
            #         interpolation=cv2.INTER_AREA,
            #     )
        imgs_to_infer = [imgs[j] for j in correct_frame_idx]
        if len(imgs_to_infer) == 0:
            return []
        results = self.model(
            source=imgs_to_infer,
            device=self.device,
            imgsz=self.cfg["inference_imgsz"],
            conf=self.cfg["inference_conf"],
            stream=False,
            verbose=False,
            half=True,
        )
        dets = [[] for _ in range(len(imgs))]
        for idx, det in zip(correct_frame_idx, results):
            dets[idx] = det.boxes.data.cpu().numpy()
        end_time_ns = perf_counter_ns()
        time_spent_ns = end_time_ns - start_time_ns
        time_spent_ms = time_spent_ns / 1e6
        if self.n_calls % self.time_logging_period == 0:
            logger.info(
                f"Detector inference on {len(correct_frame_idx)} images took {time_spent_ms:.1f} ms"
            )
        return dets