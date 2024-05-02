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

try: 
    from tensor_stream import TensorStreamConverter, FourCC, Planes, FrameRate
except ImportError:  # package not installed, skip
    pass


class IvideonStreamLoader(BaseStreamLoader):
    
    def __init__(
        self, 
        sources: list,
        buffer_length: Union[str, int] = 10,
        vid_fps: Union[str, int] = "auto",
        max_first_attempts_to_reconnect: int = 30,
        first_wait_time: float = 0.1,
        second_wait_time: float = 60,
        api_url: str = "",
        api_args: str = "",
        access_token: str = "",
        **kwargs
    ) -> "IvideonStreamLoader":
        
        self.api_url = api_url
        self.api_args = api_args
        self.access_token = access_token
        super(IvideonStreamLoader, self).__init__(
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
    
    def update(self, i, stream):
        """Read stream `i` frames in daemon thread."""
        link = None
        attempt = 0
        st = f"{i + 1}/{self.n}: {stream}... "
        w, h = 0, 0
        while link is None or self.caps[i] is None:
            link = self._get_ivideon_link(stream)
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
                    f"⚠️ Video stream {i} unresponsive, please check your IP camera connection."
                )
                if attempt < self.max_first_attempts_to_reconnect:
                    time.sleep(self.first_wait_time)
                else:
                    time.sleep(self.second_wait_time)
                attempt += 1
                reopen_status = cap.open(self._get_ivideon_link(self.sources[i]))  # re-open stream if signal was lost
                logger.info(
                    f"Attemp to re-open video stream {i}, result: {reopen_status}"
                )
            else:
                success, im = cap.retrieve()
                if not success:
                    im = None
                    logger.warning(
                        f"⚠️ Cannot decode image from video stream {i}. Unknown error."
                    )
            self.imgs[i].append(im)
            n += 1
        else:
            logger.info(f"End of stream {i}.")


class GPUIvideonStreamLoader(IvideonStreamLoader):
    
    def __init__(
        self, 
        sources: list,
        buffer_length: Union[str, int] = 10,
        vid_fps: Union[str, int] = "auto",
        max_first_attempts_to_reconnect: int = 30,
        first_wait_time: float = 0.1,
        second_wait_time: float = 60,
        api_url: str = "",
        api_args: str = "",
        access_token: str = "",
        cuda_device: int = 0,
    ) -> "IvideonStreamLoader":
        
        self.api_url = api_url
        self.api_args = api_args
        self.access_token = access_token
        self.cuda_device = cuda_device
        super(IvideonStreamLoader, self).__init__(
            sources,
            buffer_length,
            vid_fps,
            max_first_attempts_to_reconnect,
            first_wait_time,
            second_wait_time,
        )
        
        
    def update(self, i, stream):
        """Read stream `i` frames in daemon thread."""
        link = None
        attempt = 0
        st = f"{i + 1}/{self.n}: {stream}... "
        w, h = 0, 0
        while link is None or self.caps[i] is None:
            link = self._get_ivideon_link(stream)
            if link is None:
                if attempt < self.max_first_attempts_to_reconnect:
                    time.sleep(self.first_wait_time)
                else:
                    time.sleep(self.second_wait_time)
                attempt += 1
                continue
            try:
                cap = TensorStreamConverter(
                    stream, 
                    cuda_device=self.cuda_device,
                    max_consumers=1,
                    buffer_size=self.buffer_length,
                    framerate_mode=FrameRate.NATIVE_LOW_DELAY,
                    )
                cap.initialize()
                cap.start()
            except Exception as ex:
                logger.warning(
                    f"Video stream {i} is unresponsive on start: {ex}, reconnecting..."
                )
                continue
            self.caps[i] = cap
            w = int(self.caps[i].frame_size[0])
            h = int(self.caps[i].frame_size[1])
            if w == 0 or h == 0:
                attempt += 1
                self.caps[i] = None
                continue
            try:
                im = self.caps[i].read(
                    width=w,
                    height=h,
                    pixel_format=FourCC.RGB24,
                    planes_pos=Planes.PLANAR,
                    normalization=True
                    )  # guarantee first frame
            except Exception as err:
                logger.warning(
                    f"{st}Failed to read images from {stream}: {err}"
                )
                attempt += 1
                self.caps[i] = None
                continue
            if im.shape[2] == 0 or im.shape[1] == 0:
                attempt += 1
                self.caps[i] = None
                continue
        self.fps[i] = self.caps[i].fps
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
            try:
                im = self.caps[i].read(
                    width=w,
                    height=h,
                    pixel_format=FourCC.RGB24,
                    planes_pos=Planes.PLANAR,
                    normalization=True
                    )  # guarantee first frame
                success = True
            except:
                im = self.imgs[i][-1]
                success = False
            if not success:
                logger.warning(
                    f"⚠️ Video stream {i} unresponsive, please check your IP camera connection."
                )
                if attempt < self.max_first_attempts_to_reconnect:
                    time.sleep(self.first_wait_time)
                else:
                    time.sleep(self.second_wait_time)
                attempt += 1
                self.caps[i] = TensorStreamConverter(
                    self._get_ivideon_link(self.sources[i]), 
                    cuda_device=self.cuda_device,
                    max_consumers=1,
                    buffer_size=self.buffer_length,
                    framerate_mode=FrameRate.NATIVE_LOW_DELAY,
                    )
                try:
                    cap.initialize()
                    cap.start() # re-open stream if signal was lost
                except Exception as err:
                    self.caps[i] = None
                    attempt += 1
                    continue
                else:
                    err = True
                logger.info(
                    f"Attemp to re-open video stream {i}, result: {err}"
                )
            self.imgs[i].append(im)
            n += 1
        else:
            logger.info(f"End of stream {i}.")