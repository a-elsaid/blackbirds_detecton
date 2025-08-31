from ensemble_boxes import *
from numba import jit
import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# import pdb

PATIENCE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Function to calculate loss for every epoch
class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.epoch = 0
        self.current_trn_total = 0.0
        self.current_val_iou_total = 0.0
        self.current_val_precision_total = 0.0
        self.trn_step = 0
        self.val_step   = 0
        self.total_trn_losses = []
        self.total_val_ious = []
        self.total_val_precisions = []
        self.best_val = None
        self.patience = PATIENCE

    @property
    def log_trn_loss(self,):
        self.total_trn_losses.append(self.trn_value)

    @property
    def log_val(self,):
        self.total_val_ious.append(self.iou_value)
        self.total_val_precisions.append(self.precision_value)

    def send_trn(self, value):
        self.current_trn_total += value
        self.trn_step += 1

    def send_val_iou_precision(self, ious, precisions):
        self.current_val_iou_total += ious
        self.current_val_precision_total += precisions
        self.val_step += 1

    @property
    def set_best_val(self,):
        if self.best_val is None or self.iou_value > self.best_val: # found higher accuracy
            logger.info(f"Saving model as IOU is increased from {self.best_val} to {self.iou_value}")
            self.best_val = self.iou_value
            self.patience = PATIENCE
            return "improved"
        else:
            logger.info(f"IOU did not improve -- Patience: {self.patience} → {self.patience-1}")
            self.patience -= 1
            return "not_improved"

        if self.patience <= 0:
            logger.info(f"Early stopping as IOU did not improve for {PATIENCE} epochs")
            return "Quit"     

    @property
    def avg_trn_losses(self):
        return np.mean(self.total_trn_losses) if self.total_trn_losses else 0

    @property
    def trn_value(self):
        return self.current_trn_total / self.trn_step

    @property
    def precision_value(self):
        return float(self.current_val_precision_total) / self.val_step

    @property
    def iou_value(self):
        return float(self.current_val_iou_total) / self.val_step

    @property
    def reset(self):
        self.current_trn_total = 0.0
        self.current_val_iou_total = 0.0
        self.current_val_precision_total = 0.0
        self.trn_step = 0.0
        self.val_step = 0.0

    @property
    def plot_losses(self):
        plt.plot(self.total_trn_losses, 'b', label='Train losses')
        plt.plot(self.total_val_ious, 'r', label='Val IoU')
        plt.plot(self.total_val_precisions, 'g', label='Val Precision')
        plt.legend()
        plt.savefig(f"{self.log_dir}/progress.png")
        plt.cla()
        plt.clf()
        plt.close()

#@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area





#@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

#@jit(nopython=True)
def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


#@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision

def make_ensemble_predictions(images, models):
    images = list(image.to(DEVICE) for image in images)    
    result = []
    for net in models:
        net.eval()
        outputs = net(images)
        result.append(outputs)
    return result

def run_wbf(
        predictions, 
        image_index, 
        image_size=(230, 290), 
        iou_thr=0.55, 
        skip_box_thr=0.5, 
        weights=None
    ):
    '''
    Run Weighted Boxes Fusion (WBF) on the given predictions for a specific image.
    '''
    H, W = image_size
    B = np.array([p[image_index]['boxes'].data.cpu().numpy() for p in predictions])
    B[:,[0,2]] /= (W-1)
    B[:,[1,3]] /= (H-1)
    S = [p[image_index]['scores'].data.cpu().numpy() for p in predictions]
    L = [p[image_index]['labels'].data.cpu().numpy() for p in predictions]
    boxes, scores, labels = weighted_boxes_fusion(B, S, L, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    B[:,[0,2]] *= (W-1)
    B[:,[1,3]] *= (H-1)
    return boxes, scores, labels

def collate_fn(batch):
    return tuple(zip(*batch))