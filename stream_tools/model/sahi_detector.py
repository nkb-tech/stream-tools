import logging
from time import perf_counter_ns
from typing import Any
from dataclasses import dataclass
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


import numpy as np
import cv2


class NMS:
    """
    Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression).

    Args:
        element_crops (MakeCropsDetectThem): Object containing crop information.
        nms_threshold (float): IoU/IoS threshold for non-maximum suppression.  Dafault is 0.3.
        match_metric (str): Matching metric, either 'IOU' or 'IOS'. Dafault is IoS.
        class_agnostic_nms (bool) Determines the NMS mode in object detection. When set to True, NMS 
            operates across all classes, ignoring class distinctions and suppressing less confident 
            bounding boxes globally. Otherwise, NMS is applied separately for each class. Default is True.
        intelligent_sorter (bool): Enable sorting by area and rounded confidence parameter. 
            If False, sorting will be done only by confidence (usual nms). Dafault is True.
        sorter_bins (int): Number of bins to use for intelligent_sorter. A smaller number of bins makes
            the NMS more reliant on object sizes rather than confidence scores. Defaults to 5.

    Attributes:
        class_names (dict): Dictionary containing class names of yolo model.
        crops (list): List to store the CropElement objects.
        image (np.ndarray): Source image in BGR.
        nms_threshold (float): IOU/IOS threshold for non-maximum suppression.
        match_metric (str): Matching metric (IOU/IOS).
        class_agnostic_nms (bool) Determines the NMS mode in object detection.
        intelligent_sorter (bool): Flag indicating whether sorting by area and confidence parameter is enabled.
        sorter_bins (int): Number of bins to use for intelligent_sorter. 
        detected_conf_list_full (list): List of detected confidences.
        detected_xyxy_list_full (list): List of detected bounding boxes.
        detected_masks_list_full (list): List of detected masks.
        detected_polygons_list_full (list): List of detected polygons when memory optimization is enabled.
        detected_cls_id_list_full (list): List of detected class IDs.
        detected_cls_names_list_full (list): List of detected class names.
        filtered_indices (list): List of indices after non-maximum suppression.
        filtered_confidences (list): List of confidences after non-maximum suppression.
        filtered_boxes (list): List of bounding boxes after non-maximum suppression.
        filtered_classes_id (list): List of class IDs after non-maximum suppression.
        filtered_classes_names (list): List of class names after non-maximum suppression.
        filtered_masks (list): List of filtered (after nms) masks if segmentation is enabled.
        filtered_polygons (list): List of filtered (after nms) polygons if segmentation and
            memory optimization are enabled.
    """

    def __init__(self):
        pass

    @staticmethod
    def average_to_bound(confidences, N=10):
        """
        Bins the given confidences into N equal intervals between 0 and 1, 
        and rounds each confidence to the left boundary of the corresponding bin.

        Parameters:
        confidences (list or np.array): List of confidence values to be binned.
        N (int, optional): Number of bins to use. Defaults to 10.

        Returns:
        list: List of rounded confidence values, each bound to the left boundary of its bin.
        """
        # Create the bounds
        step = 1 / N
        bounds = np.arange(0, 1 + step, step)

        # Use np.digitize to determine the corresponding bin for each value
        indices = np.digitize(confidences, bounds, right=True) - 1

        # Bind values to the left boundary of the corresponding bin
        averaged_confidences = np.round(bounds[indices], 2) 

        return averaged_confidences.tolist()

    @staticmethod
    def intersect_over_union(mask, masks_list):
        """
        Compute Intersection over Union (IoU) scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask.
        """
        iou_scores = []
        for other_mask in masks_list:
            # Compute intersection and union
            intersection = np.logical_and(mask, other_mask).sum()
            union = np.logical_or(mask, other_mask).sum()
            # Compute IoU score, avoiding division by zero
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)
        return torch.tensor(iou_scores)

    @staticmethod
    def intersect_over_smaller(mask, masks_list):
        """
        Compute Intersection over Smaller area scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask,
                calculated over the smaller area.
        """
        ios_scores = []
        for other_mask in masks_list:
            # Compute intersection and area of smaller mask
            intersection = np.logical_and(mask, other_mask).sum()
            smaller_area = min(mask.sum(), other_mask.sum())
            # Compute IoU score over smaller area, avoiding division by zero
            ios = intersection / smaller_area if smaller_area != 0 else 0
            ios_scores.append(ios)
        return torch.tensor(ios_scores)

    def agnostic_nms(
        self,
        confidences: torch.tensor,
        boxes: torch.tensor,
        match_metric,
        nms_threshold,
        masks=None,
        intelligent_sorter=False, 
        sorter_bins=5,
        cls_indexes=None 
    ):
        """
        Apply class-agnostic non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.

        Args:
            confidences (torch.Tensor): List of confidence scores.
            boxes (torch.Tensor): List of bounding boxes.
            match_metric (str): Matching metric, either 'IOU' or 'IOS'.
            nms_threshold (float): The threshold for match metric.
            masks (list): List of masks. 
            intelligent_sorter (bool, optional): intelligent sorter 
            sorter_bins (int): Number of bins to use for intelligent_sorter. 
                A smaller number of bins makes the NMS more reliant on object 
                sizes rather than confidence scores. Defaults to 5.
            cls_indexes (torch.Tensor):  indexes from network detections corresponding
                to the defined class,  uses in case of not agnostic nms

        Returns:
            list: List of filtered indexes.
        """
        if len(boxes) == 0:
            return []

        # Extract coordinates for every prediction box present
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of every box
        areas = (x2 - x1) * (y2 - y1)

        # Sort the prediction boxes according to their confidence scores or intelligent_sorter mode
        if intelligent_sorter:
            # Sort the prediction boxes according to their round confidence scores and area sizes
            order = torch.tensor(
                sorted(
                    range(len(confidences)),
                    key=lambda k: (
                        self.average_to_bound(confidences[k].item(), sorter_bins),
                        areas[k],
                    ),
                    reverse=False,
                )
            )
        else:
            order = confidences.argsort()
        # Initialise an empty list for filtered prediction boxes
        keep = []
        order = order.to(boxes.device)
        while len(order) > 0:
            # Extract the index of the prediction with highest score
            idx = order[-1]
            # Push the index in filtered predictions list
            keep.append(idx.tolist())
            # Remove the index from the list
            order = order[:-1]
            # If there are no more boxes, break
            if len(order) == 0:
                break
            # Select coordinates of BBoxes according to the indices
            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)
            # Find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])
            # Find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1
            # Take max with 0.0 to avoid negative width and height
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            # Find the intersection area
            inter = w * h
            # Find the areas of BBoxes
            rem_areas = torch.index_select(areas, dim=0, index=order)
            if match_metric == "IOU":
                # Find the union of every prediction with the prediction
                union = (rem_areas - inter) + areas[idx]
                # Find the IoU of every prediction
                match_metric_value = inter / union

            elif match_metric == "IOS":
                # Find the smaller area of every prediction with the prediction
                smaller = torch.min(rem_areas, areas[idx])
                # Find the IoU of every prediction
                match_metric_value = inter / smaller

            else:
                raise ValueError("Unknown matching metric")

            # If masks are provided and IoU based on bounding boxes is greater than 0,
            # calculate IoU for masks and keep the ones with IoU < nms_threshold
            if masks is not None and torch.any(match_metric_value > 0):

                mask_mask = match_metric_value > 0 

                order_2 = order[mask_mask]
                filtered_masks = [masks[i] for i in order_2]

                if match_metric == "IOU":
                    mask_iou = self.intersect_over_union(masks[idx], filtered_masks)
                    mask_mask = mask_iou > nms_threshold

                elif match_metric == "IOS":
                    mask_ios = self.intersect_over_smaller(masks[idx], filtered_masks)
                    mask_mask = mask_ios > nms_threshold
                # create a tensor of indences to delete in tensor order
                order_2 = order_2[mask_mask]
                inverse_mask = ~torch.isin(order, order_2)

                # Keep only those order values that are not contained in order_2
                order = order[inverse_mask]

            else:
                # Keep the boxes with IoU/IoS less than threshold
                mask = match_metric_value < nms_threshold

                order = order[mask]
        if cls_indexes is not None:
            keep = [cls_indexes[i].item() for i in keep]
        return keep

    def __call__(self, *args, **kwargs):
        return self.nms(*args, **kwargs)

    def nms(
        self,
        detected_cls,
        detected_conf, 
        detected_xyxy, 
        match_metric, 
        nms_threshold, 
        intelligent_sorter=False,
        sorter_bins=None,
        detected_masks=None, 
    ):
        '''
            Performs Non-Maximum Suppression (NMS)
            Args:
                detected_cls_id (torch.Tensor): tensor containing the class IDs for each detected object.
                detected_conf (torch.Tensor):  tensor of confidence scores.
                detected_xyxy (torch.Tensor): tensor of bounding boxes.
                match_metric (str): Matching metric, either 'IOU' or 'IOS'.
                nms_threshold (float): the threshold for match metric.
                detected_masks (torch.Tensor):  List of masks. 
                intelligent_sorter (bool, optional): intelligent sorter 
                sorter_bins (int): Number of bins to use for intelligent_sorter.
                    A smaller number of bins makes the NMS more reliant on object 
                    sizes rather than confidence scores. Defaults to 5.
            Returns:
                List[int]: A list of indices representing the detections that are kept after applying
                    NMS for each class separately.
            Notes:
                - This method performs NMS separately for each class, which helps in
                    reducing false positives within each class.
                - If in your scenario, an object of one class can physically be inside
                    an object of another class, you should definitely use this non-agnostic nms
        '''
        all_keeps = []
        for cls in torch.unique(detected_cls):
            cls_indexes = torch.where(detected_cls==cls)[0]
            if detected_masks is not None:
                class_masks = [detected_masks[i] for i in cls_indexes]
            else:
                class_masks = None
            keep_indexes = self.agnostic_nms(
                    confidences=detected_conf[cls_indexes],
                    boxes=detected_xyxy[cls_indexes],
                    match_metric=match_metric,
                    nms_threshold=nms_threshold,
                    masks=class_masks,
                    intelligent_sorter=intelligent_sorter,
                    sorter_bins=sorter_bins,
                    cls_indexes=cls_indexes
                )
            all_keeps.extend(keep_indexes)
        return all_keeps


@dataclass
class ImageCrop:
    x_origin: int
    y_origin: int
    crop: np.ndarray
    
    def to_global(self, bboxes: np.ndarray):
        """
        bboxes = [[xtl, ytl, xbr, ybr, ...], ...]
        """
        bboxes[:, [0, 2]] += self.x_origin
        bboxes[:, [1, 3]] += self.y_origin
        return bboxes

class SAHIDetector:
    def __init__(self, cfg):
        self.model = YOLO(model=cfg["model_path"], task="detect")
        self.model.eval()
        self.device = torch.device(cfg["device"])
        self.cfg = cfg
        self.classes = self.cfg.get("classes", None)
        self.inf_size = self.cfg["inference_imgsz"]
        self.inf_conf = self.cfg.get("inference_conf", 0.25)
        self.inference_bs = self.cfg.get("inference_bs", 16)
        self.crop_w, self.crop_h = self.cfg["crop_size"]
        self.overlap_x, self.overlap_y = self.cfg["crop_overlap"]
        self.crop_step_x = self.crop_w - self.overlap_x
        self.crop_step_y = self.crop_h = self.overlap_y
        self.min_rel_window_size = self.cfg.get("min_rel_window_size", 0.5)
        self.nms_match_func = self.cfg.get("nms_match_func", "IOS")
        self.nms_match_threshold = self.cfg.get("nms_match_threshold", 0.3)
        self.nms_intelligent_sorter = self.cfg.get("nms_intelligent_sorter", True)
        self.nms_sorter_bins = self.cfg.get("nms_sorter_bins", 5)
        self.time_logging_period = self.cfg["time_logging_period"]
        self.nms = NMS()
        self.n_calls = -1
        self.verbose = self.cfg.get("verbose", False)
           
    @torch.no_grad() 
    def initialize(self):
        # Dummy inference for model warmup
        for _ in range(10):
            dummy_imgs = [
                np.random.randint(
                    low=0,
                    high=255,
                    size=(*self.inf_size, 3),
                    dtype=np.uint8,
                )
                for _ in range(self.inference_bs)
            ]
            self.model(dummy_imgs, 
                       device=self.device,
                       imgsz=self.inf_size,
                       conf=self.inf_conf,
                       stream=False,
                       verbose=self.verbose,
                       half=True,
                       classes=self.classes)
        self.n_calls = -1

    @property
    def names(self):
        return self.model.names
    
    def __call__(self, imgs: list) -> Any:
        self.n_calls += 1
        return self.inference(imgs)
    
    def _check_correct_images(self, imgs):
        correct_frame_idx = []
        for i in range(len(imgs)):
            if imgs[i] is None:
                continue
            correct_frame_idx.append(i)
        imgs_to_infer = [imgs[j] for j in correct_frame_idx]
        return imgs_to_infer, correct_frame_idx

    def _get_crops(self, img):
        h, w, _ = img.shape
        crops = []
        y_steps = list(range(0, h, self.crop_step_y))
        if h - y_steps[-1] < self.crop_step_y * self.min_rel_window_size:
            y_steps.pop()
        x_steps = list(range(0, h, self.crop_step_x))
        if h - x_steps[-1] < self.crop_step_x * self.min_rel_window_size:
            x_steps.pop()
        for i, y_origin in enumerate(y_steps):
            y_bottom = y_origin + self.crop_step_y if i+1 < len(y_steps) else h
            for j, x_origin in enumerate(x_steps):
                x_right = x_origin + self.crop_step_x if j+1 < len(x_steps) else w
                crops.append(ImageCrop(
                    x_origin=x_origin, 
                    y_origin=y_origin, 
                    crop=img[y_origin:y_bottom, x_origin:x_right]))
        return crops
    
    def _merge_crop_results(self, crops: list[ImageCrop], results):
        bboxes = []
        for crop, res in zip(crops, results):
            bboxes.append(crop.to_global(res.boxes.data.detach().clone()))
        bboxes = torch.vstack(bboxes)
        keep_idx = self.nms.nms(
            detected_cls=bboxes[:, 5],
            detected_conf=bboxes[:, 4],
            detected_xyxy=bboxes[:, 0:4],
            match_metric=self.nms_match_func,
            nms_threshold=self.nms_match_threshold,
            intelligent_sorter=self.nms_intelligent_sorter,
            sorter_bins=self.nms_sorter_bins
            )
        return bboxes[keep_idx]
            
    @torch.no_grad()
    def inference(self, imgs: list):
        start_time_ns = perf_counter_ns()
        imgs_to_infer, correct_frame_idx = self._check_correct_images(imgs)
        if len(imgs_to_infer) == 0:
            return []
        results = []
        for i, img in enumerate(imgs_to_infer):
            crops = self._get_crops(img)
            res = self.model(
                source=[cr.crop for cr in crops],
                device=self.device,
                imgsz=self.inf_size,
                conf=self.inf_conf,
                stream=False,
                verbose=self.verbose,
                half=True,
                classes=self.classes
            )
            results.append(self._merge_crop_results(crops, res))
            
        dets = [[] for _ in range(len(imgs))]
        for idx, det in zip(correct_frame_idx, results):
            dets[idx] = det
        end_time_ns = perf_counter_ns()
        time_spent_ns = end_time_ns - start_time_ns
        time_spent_ms = time_spent_ns / 1e6
        if self.n_calls % self.time_logging_period == 0:
            logger.info(
                f"SAHIDetector inference on {len(correct_frame_idx)} images took {time_spent_ms:.1f} ms"
            )
        return dets