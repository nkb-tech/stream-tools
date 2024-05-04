import logging
import time

from collections import deque
import torch
from torchaudio.io import StreamReader

logger = logging.getLogger(__name__)

from stream_tools.dataloader import BaseStreamLoader
from stream_tools.utils import yuv_to_rgb


class TorioLoader(BaseStreamLoader):

    def init_stream(self, stream: str, i: int, device: str='cuda:0') -> bool:
        """Init stream and fill the main info about it."""
        assert 'cuda' in device, f'Only cuda device now supported, got {device}.'
        success, im = False, None
        try:
            cap = StreamReader(stream)
            cap.add_video_stream(
                frames_per_chunk=1,
                buffer_chunk_size=1,
                decoder='h264_cuvid',
                decoder_option={"threads": "0", "gpu": "0"},
                hw_accel=device,
            )
            success = not cap.fill_buffer()
            (im, ) = cap.pop_chunks()
        except Exception as ex:
            logger.warning(f'Video stream {i} is unresponsive on start: {ex}, reconnecting...')
            self.attempts[i] += 1
            return success

        if not success or not torch.is_tensor(im):
            logger.warning(f'Failed to read images from {stream}, reconnecting...')
            self.attempts[i] += 1
            return success
        
        _, _, h, w = im.shape

        if w == 0 or h == 0:
            logger.warning(f'Failed to read shape of images from {stream}, reconnecting...')
            self.attempts[i] += 1
            return success

        success = True
        self.started[i] = True
        self.caps[i] = cap
        self.fps[i] = cap.get_src_stream_info(0).frame_rate
        self.frames[i] = float('inf')
        self.shape[i] = [w, h]
        buf = deque(maxlen=self.buffer_length)
        buf.append(yuv_to_rgb(im))
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
            success = not cap.fill_buffer()  # .stream() = .fill_buffer() followed by .pop_chunks()
            im = self.imgs[i][-1]  # Default to last valid image
            if not success:
                logger.warning(f'WARNING ⚠️ Video stream {i} unresponsive, please check your IP camera connection.')
                self.check_attempts(i, skip_first=False)
                self.attempts[i] += 1
                # TODO reopen video stream for torio loader
                # cap.open(source)  # re-open stream if signal was lost
            else:
                (im, ) = cap.pop_chunks()
            self.imgs[i].append(yuv_to_rgb(im))
            n += 1
        else:
            logger.info(f'End of stream {i}.')
