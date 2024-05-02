import cv2
import numpy as np
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime
from time import perf_counter_ns
import pytz
import logging
import warnings
from collections import defaultdict
from queue import Empty, Queue
from threading import Event, Thread
warnings.filterwarnings("ignore")

import boxmot as bx

from stream_tools.pipeline import BaseWorker
from stream_tools.dataloader import BaseStreamLoader
from stream_tools.model import Detector
from stream_tools.config import BaseConfig

TIMEZONE = pytz.timezone(
    "Europe/Moscow"
)  # UTC, Asia/Shanghai, Europe/Berlin

logger = logging.getLogger(__name__)

def timetz(*args):
    return datetime.now(TIMEZONE).timetuple()


class MultiTrackWorker(BaseWorker):
    _TIMEOUT = 2

    def __init__(self, 
                 start_timestamp: datetime,
                 dataloader: BaseStreamLoader,
                 detector: Detector,
                 tracker_cfg: dict,
                 device: torch.device,
                 cams_cfg: BaseConfig,
                 inf_cfg: dict,
                 send: bool = False,
                 debug: bool = False):
        self.device = device
        self.cams_cfg = cams_cfg
        self.inf_cfg = inf_cfg
        self.pipeline_time_logging_period = inf_cfg['pipeline_logging_period']
        # Streams
        logger.info(f"Initializing stream loader...")
        self.dataloader = dataloader
        self.dataloader.initialize()
        logger.info(f"Stream loader initialized")
        # Models
        self.detector = detector
        self.names = self.detector.names
        self.detector.initialize()
        logger.info(f"Detector initialized")
        # Trackers
        tracker_cfg['device'] = torch.device(tracker_cfg['device'])
        tracker_cfg['reid_weights'] = Path(tracker_cfg['reid_weights']) if tracker_cfg['reid_weights'] is not None else None
        self.trackers_time_logging_period = tracker_cfg.pop('time_logging_period')
        self.trackers = {
            i: bx.create_tracker(**tracker_cfg)
            for i in range(len(self.cams_cfg.cam_ids))
        }
        self.poses = {cam_id: dict() for cam_id in self.cams_cfg.cam_ids}
        self.trackers_n_calls = 0
        logger.info(f"Trackers initialized")
        # Debug
        self.debug = debug
        if self.debug:
            self.inf_cfg["debug"]["save_img_path"] = Path(
                self.inf_cfg["debug"]["save_img_path"]
            ) / self.format_time(start_timestamp)
            logger.info(
                f"Debug mode: ON, saving data to {self.inf_cfg['debug']['save_img_path']}"
            )
            save_img_path = self.inf_cfg["debug"]["save_img_path"]
            (save_img_path / "images").mkdir(
                exist_ok=True, parents=True
            )
            (save_img_path / "labels").mkdir(
                exist_ok=True, parents=True
            )
            for cam_id in self.cams_cfg.cam_ids:
                (save_img_path / "crops" / str(cam_id)).mkdir(exist_ok=True, parents=True)
            self.save_img_path = save_img_path
        else:
            logger.info(f"Debug mode: OFF")

        super(MultiTrackWorker, self).__init__(send, debug)

    def pipeline(self, imgs, timestamp):
        dets = self.detector(imgs)
        track_res = self.run_trackers(dets, imgs)
        results = {
            'tracks': track_res,
            'dets': dets,
        }
        return results

    def run_trackers(self, dets, imgs):
        # TODO Check tracking vs detection results
        start_time_ns = perf_counter_ns()
        track_res = defaultdict(list)
        n_dets = 0
        for (i, tracker), det, img in zip(self.trackers.items(), dets, imgs):
            if len(det) == 0:
                det = np.empty((0, 6))
            if img is None:
                continue
            n_dets += len(det)
            tracks = tracker.update(det, img)
            img_h, img_w, _ = img.shape
            if tracks.shape[0] != 0:
                tracks[:, [0,2]] = np.clip(tracks[:, [0,2]], 0, img_w)
                tracks[:, [1,3]] = np.clip(tracks[:, [1,3]], 0, img_h)
                track_res[i] = tracks # [xtl, ytl, xbr, ybr, id, conf, class, index (from detections)]
        end_time_ns = perf_counter_ns()
        time_spent_ns = end_time_ns - start_time_ns
        time_spent_ms = time_spent_ns / 1e6
        if self.trackers_n_calls % self.trackers_time_logging_period == 0:
            n_imgs = len([i for i in imgs if i is not None])
            logger.info(
                f"Trackers inference on {n_imgs} images and {n_dets} detections took {time_spent_ms:.1f} ms"
            )
        self.trackers_n_calls += 1
        return track_res
                            

    def log_debug(self, timestamp, results, imgs):
        # TODO rewrite
        timestamp_str = self.format_time(timestamp)
        tracks = results['tracks']
        dets = results['dets']
        for i, (tracks, cam_idx) in enumerate(tracks):
            img = imgs[cam_idx]
            img_h, img_w, _ = img.shape
            try:
                cv2.imwrite(
                    str(
                        self.save_img_path
                        / "images"
                        / f"{self.cams_cfg.cam_ids[cam_idx]}_{timestamp_str}.jpg"
                    ),
                    img,
                )
            except Exception as err:
                logger.error(f'Error in image saving: {img.shape}, {err}', exc_info=True)
            labels_str = []
            for track in tracks: # [xc, yc, wn, hn, id, conf, class, index (from detections)]
                xcn, ycn, wn, hn, id_obj, conf, label, ind = track
                crop = img[
                    int((ycn - hn)*img_h):int((ycn + hn)*img_h),
                    int((xcn - wn)*img_w):int((xcn + wn)*img_w),
                ]
                labels_str.append(f'{int(label)} {xcn} {ycn} {wn} {hn} {conf}\n')
                Path(self.save_img_path
                     / 'crops' 
                     / f'{self.cams_cfg.cam_ids[cam_idx]}'
                     / f'{self.names[label]}_{id_obj}').mkdir(exist_ok=True, parents=True)
                try: 
                    cv2.imwrite(
                        str(
                            self.save_img_path
                            / 'crops' 
                            / f'{self.cams_cfg.cam_ids[cam_idx]}'
                            / f'{self.names[label]}_{id_obj}'
                            / f'{timestamp_str}.jpg'
                        ),
                        crop
                    )
                except Exception as err:
                    logger.error(f'Error in image saving: {crop.shape}, {track}, {img.shape}, {err}', exc_info=True)
            with (
                self.save_img_path
                / "labels"
                / f"{self.cams_cfg.cam_ids[cam_idx]}_{timestamp_str}.txt"
            ).open("w") as f:
                f.writelines(labels_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", type=str, help="Inference config path")
    parser.add_argument("-cam", type=str, help="Camera config path")
    parser.add_argument(
        "-log", "--log_path", type=str, help="Logging path"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode, that save images and predictions",
    )
    args = parser.parse_args()
    cfg_path = args.cfg
    cams = args.cam
    log_path = args.log_path
    debug = args.debug
    start_timestamp = datetime.now()
    Path(log_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/{TIMEZONE.localize(start_timestamp).isoformat('T', 'seconds').replace(':', '_')}_logs.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(cams, "r") as f:
        cams_cfg = yaml.safe_load(f)
    worker = MultiTrackWorker(start_timestamp, cfg, cams_cfg, debug)
    worker.run()
    

if __name__ == '__main__':
    main()
        