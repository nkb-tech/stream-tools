import logging
import os
import time
from collections import deque

import cv2

logger = logging.getLogger(__name__)

from stream_tools.dataloader import BaseStreamLoader


class OpenCVLoader(BaseStreamLoader):

    def update(self, i, source):
        """Read stream `i` frames in daemon thread."""
        # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;2"
        attempt = 0
        st = f'{i + 1}/{self.n}: {source}... '
        w, h = 0, 0
        while self.caps[i] is None:
            if attempt == 0:
                pass
            elif attempt < self.max_first_attempts_to_reconnect:
                time.sleep(self.first_wait_time)
            else:
                time.sleep(self.second_wait_time)
            try:
                cap = cv2.VideoCapture(source)
            except Exception as ex:
                logger.warning(f'Video stream {i} is unresponsive on start: {ex}, reconnecting...')
                attempt += 1
                continue
            self.caps[i] = cap
            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                logger.warning(f'{st}Failed to read images from {source}, reconnecting...')
                attempt += 1
                self.caps[i] = None
                continue
            w = int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w == 0 or h == 0:
                attempt += 1
                self.caps[i] = None
                continue
        self.fps[i] = self.caps[i].get(cv2.CAP_PROP_FPS)
        self.frames[i] = float('inf')
        self.shape[i] = [w, h]
        buf = deque(maxlen=self.buffer_length)
        buf.append(im)
        self.imgs[i] = buf
        logger.info(f'{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')
        attempt = 0
        n, f = 0, self.frames[i]  # frame number, frame array
        self.started[i] = 1
        while self.running and n < (f - 1):  # and cap.isOpened()
            success = (cap.grab())  # .read() = .grab() followed by .retrieve()
            im = self.imgs[i][-1]  # Default to last valid image
            if not success:
                logger.warning(f'WARNING ⚠️ Video stream {i} unresponsive, please check your IP camera connection.')
                if attempt < self.max_first_attempts_to_reconnect:
                    time.sleep(self.first_wait_time)
                else:
                    time.sleep(self.second_wait_time)
                attempt += 1
                cap.open(source)  # re-open stream if signal was lost
            else:
                success, im = cap.retrieve()
                attempt = 0
                if not success:
                    im = None
                    logger.warning(f'WARNING ⚠️ Cannot decode image from video stream {i}. Unknown error.')
                    if attempt < self.max_first_attempts_to_reconnect:
                        time.sleep(self.first_wait_time)
                    else:
                        time.sleep(self.second_wait_time)
                    attempt += 1
                    cap.open(source)
            self.imgs[i].append(im)
            n += 1
        else:
            logger.info(f'End of stream {i}.')
