import argparse
from pathlib import Path
from datetime import datetime
import pytz
import logging
import warnings
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter_ns
warnings.filterwarnings("ignore")

TIMEZONE = pytz.timezone(
    "Europe/Moscow"
)  # UTC, Asia/Shanghai, Europe/Berlin

logger = logging.getLogger(__name__)

def timetz(*args):
    return datetime.now(TIMEZONE).timetuple()


class BaseWorker:
    _TIMEOUT = 2

    def __init__(self, 
                 send: bool = False,
                 debug: bool = False):
        # Init separate process
        self.queue = Queue(maxsize=1)
        self.done = Event()
        self.pool = Thread(target=self.worker, daemon=True)
        self.debug = debug
        self.send = send
        self.pipeline_n_calls = 0
        # self.pipeline_time_logging_period = 1000 # default, overwrite from config

        # Streams
        pass
        # Models
        pass
        # Trackers
        pass
        # Debug
        pass

        self.pool.start()
    
    @staticmethod
    def format_time(timestamp):
        return timestamp.isoformat("T", "milliseconds").replace(
            ":", "_"
        ).replace('.', '_')

    def worker(self) -> None:
        while not self.done.is_set():
            try:
                imgs = self.queue.get(
                    block=True,
                    timeout=self._TIMEOUT,
                )
                try:
                    self.run_on_images(imgs)
                except Exception as err:
                    logger.error(f'Error in work: {err}', exc_info=True)
                finally:
                    self.queue.task_done()
            except Empty as e:
                pass
        return

    def __del__(self):
        self.pool.signal_exit()

    def run(self):
        for imgs in self.dataloader:
            self.queue.put(imgs)
            # if self.pipeline_n_calls % 10 == 0:
            #     logger.info(f'Queue length: {self.queue.qsize()}')

    def run_on_images(self, imgs):
        timestamp = datetime.now(TIMEZONE)
        try:
            start_time_ns = perf_counter_ns()
            results = self.pipeline(imgs, timestamp)
            if self.debug:
                self.log_debug(timestamp, results, imgs)
            if self.send:
                self.send_results(timestamp, results, imgs)
            end_time_ns = perf_counter_ns()
            time_spent_ns = end_time_ns - start_time_ns
            time_spent_ms = time_spent_ns / 1e6
            if self.pipeline_n_calls % self.pipeline_time_logging_period == 0:
                n_imgs = len([i for i in imgs if i is not None])
                logger.info(
                    f"Pipeline inference on {n_imgs} images took {time_spent_ms:.1f} ms"
                )
            self.pipeline_n_calls += 1
        except Exception as err:
            logger.error(f'Error in pipeline: {err}', exc_info=True)
            import ipdb; ipdb.set_trace() # For debugging only
            results = self.pipeline(imgs, timestamp)
            if self.debug:
                self.log_debug(timestamp, results, imgs)
            if self.send:
                self.send_results(timestamp, results, imgs)
            end_time_ns = perf_counter_ns()
    
    def pipeline(self, imgs):
        raise NotImplementedError

    def log_debug(self, timestamp, results, imgs):
        raise NotImplementedError
    
    def send_results(timestamp, results, imgs):
        raise NotImplementedError


def base_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--send",
        action="store_true",
        help="Send results via API or put them into DB",
    )
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
    send = args.send
    log_path = args.log_path
    debug = args.debug
    Path(log_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/{TIMEZONE.localize(datetime.now()).isoformat('T', 'seconds').replace(':', '_')}_logs.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    worker = BaseWorker(send, debug)
    worker.run()
    

if __name__ == '__main__':
    base_main()
        