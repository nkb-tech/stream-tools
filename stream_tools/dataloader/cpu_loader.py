#!/usr/bin/env python

import logging
import time
from collections import deque
import requests as re
import json
from typing import Union

import cv2

from stream_tools.dataloader import BaseStreamLoader

logger = logging.getLogger(__name__)


class CPUStreamLoader(BaseStreamLoader):
    
    def __init__(
        self, 
        sources: list,
        source_types: list,
        buffer_length: Union[str, int] = 10,
        vid_fps: Union[str, int] = "auto",
        max_first_attempts_to_reconnect: int = 30,
        first_wait_time: float = 0.1,
        second_wait_time: float = 60,
        **kwargs
    ) -> "CPUStreamLoader":
        
        self.api_url = kwargs.get('api_url', None)
        self.api_args = kwargs.get('api_args', None)
        self.access_token = kwargs.get('access_token', None)
        self.source_types = source_types
        super(CPUStreamLoader, self).__init__(
            sources,
            buffer_length,
            vid_fps,
            max_first_attempts_to_reconnect,
            first_wait_time,
            second_wait_time,
        )
    
    def _get_ivideon_link(self, s):
        url = f'{self.api_url}cameras/{s}/{self.api_args}&access_token={self.access_token}'
        resp = re.get(url)
        if resp.status_code != 200:
            logger.warning(f'Could not get stream url from camera {s}\n{resp.text}')
            return None
        stream_url = json.loads(resp.text)['result']['url']
        return stream_url
    
    def _get_link(self, i):
        match self.source_types[i]:
            case 'ivideon':
                return self._get_ivideon_link(self.sources[i])
            case 'rtsp':
                return self.sources[i]
            case _:
                raise ValueError(f'Stream type {self.source_types[i]} is not supported')
    
    def update(self, i, stream):
        """Read stream `i` frames in daemon thread."""
        link = None
        attempt = 0
        st = f"{i + 1}/{self.n}: {stream}... "
        w, h = 0, 0
        while link is None or self.caps[i] is None:
            link = self._get_link(i)
            if link is None:
                if attempt < self.max_first_attempts_to_reconnect:
                    time.sleep(self.first_wait_time)
                else:
                    time.sleep(self.second_wait_time)
                attempt += 1
                continue
            try:
                cap = cv2.VideoCapture(link)
            except Exception as ex:
                logger.warning(
                    f"Video stream {i} is unresponsive on start: {ex}, reconnecting..."
                )
                continue
            self.caps[i] = cap
            success, im = self.caps[i].read()  # guarantee first frame
            if not success or im is None:
                logger.warning(
                    f"{st}Failed to read images from {stream}"
                )
                attempt += 1
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
        logger.info(
                f"{st}Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)"
            )
        
        n, f = 0, self.frames[i]  # frame number, frame array
        while self.running and n < (f - 1):
            success = (
                cap.grab()
            )  # .read() = .grab() followed by .retrieve()
            im = self.imgs[i][-1]
            if not success:
                logger.warning(
                    f"WARNING ⚠️ Video stream {i} unresponsive, please check your IP camera connection."
                )
                if attempt < self.max_first_attempts_to_reconnect:
                    time.sleep(self.first_wait_time)
                else:
                    time.sleep(self.second_wait_time)
                attempt += 1
                reopen_status = cap.open(self._get_link(i))  # re-open stream if signal was lost
                logger.info(
                    f"Attemp to re-open video stream {i}, result: {reopen_status}"
                )
            else:
                success, im = cap.retrieve()
                if not success:
                    im = None
                    logger.warning(
                        f"WARNING ⚠️ Cannot decode image from video stream {i}. Unknown error."
                    )
            self.imgs[i].append(im)
            n += 1
        else:
            logger.info(f"End of stream {i}.")