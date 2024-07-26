import logging
from time import perf_counter_ns
from typing import Any
import io

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, cfg):
        if 'encryption_key' in cfg.keys():
            # Initialize model from non-trained placeholder
            self.model = YOLO(model=cfg["placeholder"], task="detect")
            weights = self.decrypt_model(cfg["model_path"], cfg["encryption_key"])
            state_dict = torch.load(weights, torch.device(cfg['device']))
            self.model.load_state_dict(
                            state_dict,
                            strict=True)            
        # if cfg.get('model_type') is not None:
        #     if cfg['model_type'] == 'torchscript':
        #         raise NotImplementedError
        #         if 'encryption_key' in cfg.keys():
        #             model = self.decrypt_model(cfg["model_path"], cfg["encryption_key"])
        #         else:
        #             model = cfg['model_path']
        #         self.model = torch.load(model)
        #     elif cfg['model_type'] == 'yolo_raw':
        #         raise NotImplementedError
        #         if cfg.get('model_cfg') is None:
        #             raise KeyError(
        #                 "Config must have model_cfg as in https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml"
        #             )
        #         if 'encryption_key' in cfg.keys():
        #             weights = self.decrypt_model(cfg["model_path"], cfg["encryption_key"])
        #         else:
        #             with open(cfg["model_path"], 'rb') as model_file:
        #                 weights = io.BytesIO(model_file.read())
        #         model = DetectionModel(cfg['model_cfg'])
        #         model.load_state_dict(
        #             torch.load(weights, cfg['device']),
        #             strict=True)

            # else:
                # self.model = YOLO(model=cfg["model_path"], task="detect")
            # self.model.load_state_dict(
            #     torch.load(io.BytesIO(decrypted_model),
            #             map_location=torch.device(cfg['device'])),
            #     strict=False)
        else:
            self.model = YOLO(model=cfg["model_path"], task="detect")
        self.device = torch.device(cfg['device'])
        self.cfg = cfg
        self.classes = self.cfg.get('classes', None)
    
    def decrypt_model(self, model_path, encryption_key):
        from cryptography.fernet import Fernet
        with open(model_path, 'rb') as model_file:
            model_bytes = model_file.read()
        cipher = Fernet(encryption_key)
        decrypted_model = cipher.decrypt(model_bytes)
        return io.BytesIO(decrypted_model)

    def initialize(self):
        # Dummy inference for model warmup
        for _ in range(5):
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
                half=False,
                classes=self.classes
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
            correct_frame_idx.append(i)
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
            classes=self.classes
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