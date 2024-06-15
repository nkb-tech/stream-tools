import logging
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Transforms:
    def __init__(
        self,
        transforms: A.Compose,
    ) -> None:
        self.transforms = transforms

    def __call__(
        self,
        img,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.transforms(image=np.array(img))["image"]


class BaseClassifier:
    def __call__(self, imgs: Any) -> Any:
        self.n_calls += 1
        start_time_ns = perf_counter_ns()
        
        res = self.inference(imgs)
        
        end_time_ns = perf_counter_ns()
        time_spent_ns = end_time_ns - start_time_ns
        time_spent_ms = time_spent_ns / 1e6
        if self.n_calls % self.time_logging_period == 0:
            logger.info(
                f"Classifier inference on {len(imgs)} images took {time_spent_ms:.1f} ms"
            )
        return res
    
    def inference(self, imgs: Any) -> Any:
        raise NotImplementedError()


class YoloClassifier(BaseClassifier):
    def __init__(self, cfg, device):
        self.model = YOLO(model=cfg["model_path"], task="classify")
        self.device = device
        self.cfg = cfg
        self.time_logging_period = cfg["time_logging_period"]
        self.n_calls = -1
        super(YoloClassifier, self).__init__()

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
                # conf=cfg["inference_conf"],
                stream=False,
                verbose=False,
                half=True,
            )

    @property
    def names(self):
        return self.model.names

    def inference(self, imgs: Any):
        correct_frame_idx = []
        if type(imgs) != list:
            imgs = [imgs]
            single = True
        else:
            single = False
        for i in range(len(imgs)):
            if imgs[i] is None:
                continue
            h, w, _ = imgs[i].shape
            correct_frame_idx.append(i)
            if (h, w) != (
                self.cfg["orig_img_h"],
                self.cfg["orig_img_w"],
            ):
                imgs[i] = cv2.resize(
                    imgs[i],
                    (self.cfg["orig_img_w"], self.cfg["orig_img_h"]),
                    # Default YOLO interpolation
                    interpolation=cv2.INTER_AREA,
                )
        imgs_to_infer = [imgs[j] for j in correct_frame_idx]
        results = self.model(
            source=imgs_to_infer,
            device=self.device,
            imgsz=self.cfg["inference_imgsz"],
            # conf=self.cfg["inference_conf"],
            stream=False,
            verbose=False,
            half=True,
        )
        preds = [None for _ in range(len(imgs))]
        confs = [None for _ in range(len(imgs))]
        for idx, pred in zip(correct_frame_idx, results):
            preds[idx] = int(pred.probs.top1)
            confs[idx] = float(pred.probs.top1conf)
        return (preds[0], confs[0]) if single else (preds, confs)
    

class TorchClassifier(BaseClassifier):
    def __init__(self, cfg, device):
        # TODO add zones
        self.model = torch.jit.load(cfg["model_path"], map_location=device)
        self.model.eval()
        self.device = device
        self.cfg = cfg
        self.idx2class = cfg['idx2class']
        self.time_logging_period = self.cfg["time_logging_period"]
        self.n_calls = -1
        match cfg['resize_mode']:
            case 'resize':
                resizing = A.Resize(
                    cfg['img_size'], 
                    cfg['img_size'], 
                    interpolation=cv2.INTER_AREA,
                    always_apply=True
                )
            case 'pad':
                resizing = A.Sequential(
                    [
                        A.LongestMaxSize(
                            cfg['img_size'], 
                            always_apply=True
                        ),
                        A.PadIfNeeded(
                            cfg['img_size'],
                            cfg['img_size'],
                            always_apply=True,
                            border_mode=cv2.BORDER_CONSTANT,
                        )
                    ]
                )
        self.pipeline = Transforms(
            A.Compose(
                [
                    resizing,
                    A.Normalize(
                        **cfg['normalization_coefs'],
                        always_apply=True
                    ),
                    ToTensorV2(),
                ]
            )
        )
        super(TorchClassifier, self).__init__()
    
    @torch.no_grad() 
    def initialize(self):
        # Dummy inference for model warmup
        for _ in range(100):
            dummy_imgs = [
                np.random.randint(
                    low=0,
                    high=255,
                    size=(self.cfg["img_size"], self.cfg["img_size"], 3),
                    dtype=np.uint8,
                )
                for _ in range(self.cfg["inference_bs"])
            ]
            tensor = torch.stack([self.pipeline(im) for im in dummy_imgs]).to(self.device)
            self.model(tensor)
    
    @property
    def names(self):
        return self.idx2class.values()

    @torch.no_grad()
    def inference(self, imgs: Any):
        correct_frame_idx = []
        if type(imgs) != list:
            imgs = [imgs]
            single = True
        else:
            single = False
        tensors = [None] * len(imgs)
        for i in range(len(imgs)):
            if imgs[i] is None:
                continue
            tensors[i] = self.pipeline(imgs[i])
            correct_frame_idx.append(i)
        
        tensors_to_infer = torch.stack(
            [tensors[j] for j in correct_frame_idx]
        ).to(self.device)
        results = self.model(tensors_to_infer)
        sftmx = F.softmax(results, dim=-1).cpu().numpy()
        classes = results.argmax(dim=-1).cpu().numpy()
        preds = [None for _ in range(len(imgs))]
        confs = [None for _ in range(len(imgs))]
        for idx, pred, conf in zip(correct_frame_idx, classes, sftmx):
            preds[idx] = int(pred)
            confs[idx] = conf
        
        return (preds[0], confs[0]) if single else (preds, confs)