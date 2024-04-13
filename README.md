# Stream tools

A repo with tools for streaming models pipelines.

## Installation

```bash
git clone git@github.com:nkb-tech/stream-tools.git stream_tools
cd stream_tools
pip install .
```

## Usage example

```python
from stream_tools.dataloader import OpenCVLoader
from stream_tools.model import Detector

sources = [
    'rtsp://stream1',
    'rtsp://stream2',
    ...
]
detector_config = {
    'model_path': 'path/to/yolov8.pt',
    'inference_imgsz': 960,
    'inference_conf': 0.5,
    'inference_bs': len(sources),
    'orig_img_h': 720,
    'orig_img_w': 1280,
    'device': 'cuda:0',
    'time_logging_period': 100,
}
detector = Detector(detector_config)
detector.initialize()

loader = OpenCVLoader(
    sources,
    buffer_length=2,
    vid_fps=10,
    ...
    )
loader.initialize()
for imgs in loader:
    predicts = detector(imgs)
    ...

```

## Modules

### Dataloader

`stream_tools.dataloder` contains multithreaded rtsp-stream dataloaders, based on `opencv-python` (for CPU video decoding) and [tensor-stream](https://github.com/osai-ai/tensor-stream) (for GPU video decoding) packages.
Represented classes are: `OpenCVLoader`, `IvideonStreamLoader`, `GPUIvideonStreamLoader`. To create new dataloader, use `BaseStreamLoader` as a base class, it has all the multithreading magic in it. You will only need to implement the `update` method to start.

### Model

`stream_tools.model` has an [ultralytics](https://github.com/ultralytics/ultralytics) detection (`Detector`) and classification (`Classifier`) model wrappers with warmup, logging and batching built in.

### Pipeline

`stream_tools.pipeline` has several worker classes for mutlicamera detection and tracking tasks. Check out the `MultiTrackWorker` class. For further development, use `BaseWorker` as a starter class.

```python
from stream_tools.dataloader import OpenCVLoader
from stream_tools.model import Detector
from stream_tools.pipeline import MultiTrackWorker

sources = [
    'rtsp://stream1',
    'rtsp://stream2',
    ...
]
detector_config = {
    'model_path': 'path/to/yolov8.pt',
    'inference_imgsz': 960,
    'inference_conf': 0.5,
    'inference_bs': len(sources),
    'orig_img_h': 720,
    'orig_img_w': 1280,
    'device': 'cuda:0',
    'time_logging_period': 100,
}
detector = Detector(detector_config)

loader = OpenCVLoader(
    sources,
    buffer_length=2,
    vid_fps=10,
    ...
    )

worker = MultiTrackWorker(
    dataloader=dataloader,
    detector=detector,
    tracker_cfg=tracker_cfg, # will be added soon
    device=torch.device('cuda:0'),
    cams_cfg=cams_cfg, # will be added soon
    inf_cfg=inf_cfg, # will be added soon
    send=False,
    debug=True
)
worker.run()

```
