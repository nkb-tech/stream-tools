import logging
import time
from threading import Thread
from typing import Union

import cv2

logger = logging.getLogger(__name__)


class BaseStreamLoader:
    """BaseStreamLoader, i.e. `#RTSP, RTMP, HTTP streams`."""

    def __init__(self,
                 sources: list,
                 buffer_length: Union[str, int] = 10,
                 vid_fps: Union[str, int] = 'auto',
                 max_first_attempts_to_reconnect: int = 30,
                 first_wait_time: float = 0.1,
                 second_wait_time: float = 60,
                 **kwargs) -> 'BaseStreamLoader':
        """Initialize stream loading threads from sources according to arguments
        Args:
            sources: a list of links to video streams
            buffer_length: number of last images keep in buffer from each stream
            vid_fps: New FPS for all videos. if 'auto', all videos will be aligned by min fps in streams.
                     If `isinstance(vid_fps, int)` - all videos will be set to this value.
                     If vid_fps == -1, fps alignment will not be executed.
            max_first_attempts_to_reconnect: number of attempts to fastly reconnect to streams
            first_wait_time: fast reconnect wait time
            second_wait_time: slow reconnect wait time (if the stream is down for a long time)
        """
        # Save arguments
        self.buffer_length = buffer_length  # max buffer length
        self.vid_fps = vid_fps
        self.max_first_attempts_to_reconnect = (max_first_attempts_to_reconnect)
        self.first_wait_time = first_wait_time
        self.second_wait_time = second_wait_time
        self.running = True  # running flag for Thread
        self.sources = sources
        self.n = len(sources)
        self.kwargs = kwargs
        # Initilize attributes
        self.imgs = [None] * self.n  # buffer with images
        self.fps = [float('inf')] * self.n  # fps of each stream
        self.frames = [0] * self.n  # number of frames in each stream
        self.threads = [None] * self.n  # buffer stored streams
        self.shape = [[] for _ in range(self.n)]
        self.caps = [None] * self.n  # video capture objects
        self.started = [0] * self.n

    def initialize(self):
        # Create a thread for each source and start it
        for i, s in enumerate(self.sources):  # index, source
            # Start thread to read frames from video stream
            self.threads[i] = Thread(
                target=self.update,
                args=([i, s]),
                daemon=True,
            )
            self.threads[i].start()
        self.new_fps = (min(self.fps) if isinstance(self.vid_fps, str) and self.vid_fps == 'auto' else self.vid_fps
                        )  # fps alignment
        logger.info('')  # newline

    @property
    def bs(self):
        return self.__len__()

    def add_source(self, source: str):
        i = len(self.threads)
        self.imgs.append(None)
        self.fps.append(float('inf'))
        self.frames.append(0)
        # self.threads.append(None)
        self.shape.append([])
        self.caps.append(None)
        self.started.append(0)
        self.threads.append(Thread(
            target=self.update,
            args=([i, source]),
            daemon=True,
        ))
        self.threads[i].start()
        return i

    def close_source(self, source: Union[str, int]):
        # TODO check source and finish func
        self.threads = self.threads[:source] + self.threads[source + 1:]
        self.imgs = self.imgs[:source] + self.imgs[source + 1:]
        self.imgs = self.imgs[:source] + self.imgs[source + 1:]
        self.imgs = self.imgs[:source] + self.imgs[source + 1:]
        if self.threads[source].is_alive():
            self.threads[source].join(timeout=5)  # Add timeout

    def update(self, i, source):
        raise NotImplementedError('Implement update function in stream loader class')

    def close(self):
        """Close stream loader and release resources."""
        self.running = False  # stop flag for Thread
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)  # Add timeout
        for (cap) in (self.caps):  # Iterate through the stored VideoCapture objects
            try:
                cap.release()  # release video capture
            except Exception as e:
                logger.warning(f'WARNING ⚠️ Could not release VideoCapture object: {e}')
        # cv2.destroyAllWindows()

    def __iter__(self):
        """Iterates through image feed and re-opens unresponsive streams."""
        self.count = -1
        return self

    def __next__(self):
        """Returns original images for processing."""
        self.count += 1
        images = []

        # sleep to align fps
        time.sleep(1 / self.new_fps)

        for i, x in enumerate(self.imgs):
            # If image is not available
            if not x:
                if not self.threads[i].is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
                    self.close()
                    raise StopIteration
                # logger.warning(f"WARNING ⚠️ Waiting for stream {i}")
                im = None
            # Get the last element from buffer
            else:
                # Main process just read from buffer, not delete
                im = x[-1]

            images.append(im)

        return images

    def __len__(self):
        """Return the length of the sources object."""
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years
