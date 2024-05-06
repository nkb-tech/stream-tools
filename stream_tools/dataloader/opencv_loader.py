import logging
from collections import deque

import cv2

logger = logging.getLogger(__name__)

from stream_tools.dataloader import BaseStreamLoader


class OpenCVLoader(BaseStreamLoader):

    def init_stream(self, stream: str, i: int, device: str = 'cpu') -> bool:
        """Init stream and fill the main info about it."""
        assert 'cpu' in device, f'Only cpu device now supported, got {device}.'
        success, im = False, None
        try:
            cap = cv2.VideoCapture(stream)
            success, im = cap.read()  # guarantee first frame
        except Exception as ex:
            logger.warning(f'Video stream {i} is unresponsive on start: {ex}, reconnecting...')
            self.attempts[i] += 1
            return success

        if not success or im is None:
            logger.warning(f'Failed to read images from {stream}, reconnecting...')
            self.attempts[i] += 1
            return success

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w == 0 or h == 0:
            logger.warning(f'Failed to read shape of images from {stream}, reconnecting...')
            self.attempts[i] += 1
            return success

        success = True
        self.started[i] = True
        self.caps[i] = cap
        self.fps[i] = cap.get(cv2.CAP_PROP_FPS)
        self.frames[i] = float('inf')
        self.shape[i] = [w, h]
        buf = deque(maxlen=self.buffer_length)
        buf.append(im)
        self.imgs[i] = buf

        return success

    def update(self, i: int, source: str) -> None:
        """Read stream `i` frames in daemon thread."""
        self.attempts[i] = 0
        st = f'{i + 1}/{self.n}: {source}... '

        # main init stream loop (can be infinity)
        while not self.init_stream(source, i):
            self.check_attempts(i, skip_first=True)

        logger.info(
            f'{st}Success ✅ ({self.frames[i]} frames of shape {self.shape[i][0]}x{self.shape[i][1]} at {self.fps[i]:.2f} FPS)'
        )
        self.attempts[i] = 0
        n, f = 0, self.frames[i]  # frame number, frame array
        cap = self.caps[i]
        while self.running and n < (f - 1):  # and cap.isOpened()
            success = cap.grab()  # .read() = .grab() followed by .retrieve()
            im = self.imgs[i][-1]  # Default to last valid image
            if not success:
                logger.warning(f'WARNING ⚠️ Video stream {i} unresponsive, please check your IP camera connection.')
                self.check_attempts(i, skip_first=False)
                self.attempts[i] += 1
                cap.open(source)  # re-open stream if signal was lost
            else:
                success, im = cap.retrieve()
                self.attempts[i] = 0
                if not success:
                    im = None
                    logger.warning(f'WARNING ⚠️ Cannot decode image from video stream {i}. Unknown error.')
                    self.check_attempts(i, skip_first=False)
                    self.attempts[i] += 1
                    cap.open(source)
            self.imgs[i].append(im)
            n += 1
        else:
            logger.info(f'End of stream {i}.')
