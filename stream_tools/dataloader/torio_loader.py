import logging
from collections import deque

import torch
from torchaudio.io import StreamReader
from torchaudio.utils import ffmpeg_utils

logger = logging.getLogger(__name__)

from stream_tools.dataloader import BaseStreamLoader
from stream_tools.utils import yuv_to_rgb


class TorioLoader(BaseStreamLoader):

    def init_stream(
        self,
        stream: str,
        i: int,
        device: str = 'cuda:0',
        decoder: str = 'h264',
    ) -> bool:
        """Init stream and fill the main info about it."""
        success, im = False, None
        decode_config = {
            'frames_per_chunk': 1,
            'buffer_chunk_size': 1,
            'decoder': decoder,
            'decoder_option': {
                'threads': '0', }, }
        if 'cuda' in device:
            decode_config['decoder'] = f'{decoder}_cuvid'
            decode_config['hw_accel'] = device
            decode_config['decoder_option']['gpu'] = '0'

        assert decode_config['decoder'] in ffmpeg_utils.get_video_decoders().keys(), \
            f'Decoder {decoder} is not supported. Please check available decoder.'
        try:
            cap = StreamReader(stream)
            cap.add_video_stream(**decode_config)
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
        for (im, ) in cap.stream():
            if not self.running or n < (f - 1):
                break
            self.imgs[i].append(yuv_to_rgb(im))
            n += 1

        logger.info(f'End of stream {i}.')
