from loguru import logger
import torch
from tqdm import tqdm
from typing import List
import numpy as np
from helper import run_wbf, make_ensemble_predictions, calculate_image_precision
import datetime
from time import time
import cv2 as cv

def _train_one_epoch(model, data_loader, optimizer, device, loss_hist) -> float:
    model.train()
    for i, (images, targets, _) in enumerate(tqdm(data_loader, total=data_loader.__len__(), desc="train epoch", leave=False)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if k =='labels' else v.float().to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_hist.send_trn(losses.item())
    loss_hist.log_trn_loss

@torch.no_grad()
def evaluate(
            models, 
            data_loader, 
            device, 
            loss_hist, 
            iou_thresholds: List[float] | None = None, 
            burn_on_dir: str | None = None,
    ) -> None:
    for model in models: model.eval()
    if iou_thresholds is None: iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
    
    classes_by_names = models[0].classes
    classes_by_num = {v: k for k, v in classes_by_names.items()}

    val_image_precisions = []
    val_image_ious = []
    for images, targets, _ in tqdm(data_loader, total=data_loader.__len__(), desc="val epoch", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if k =='labels' else v.float().to(device) for k, v in t.items()} for t in targets]
        preds = make_ensemble_predictions(images, models)
        for i, img in enumerate(images):
            boxes, scores, labels = run_wbf(preds, image_index=i, image_size=img.shape[1:3])
            boxes = boxes.astype(np.float32)
            gt_boxes = targets[i]["boxes"].detach().cpu().numpy()
            gt_labels = targets[i]["labels"].detach().cpu().numpy()
            pr_sorted_idxs = np.argsort(scores)[::-1]
            pr_sorted = boxes[pr_sorted_idxs]
            image_precision = calculate_image_precision(pr_sorted, gt_boxes, thresholds=iou_thresholds, form="pascal")
            val_image_ious.append(np.mean(scores) if scores.size > 0 else 0)  # using mean score as a proxy for iou
            val_image_precisions.append(np.mean(image_precision) if image_precision.size > 0 else 0)

    
            if burn_on_dir:
                img = data_loader.dataset.__get_img__(i)
                for b, c in zip(gt_boxes, gt_labels):
                    x1,y1,x2,y2 = b.astype(int) 
                    y = y1 - 4 if y1 - 4 > 4 else y1 + 4
                    c = int(c)
                    cv.putText(img, f"{classes_by_num[c]}<<", (x1, y), cv.FONT_HERSHEY_SIMPLEX, .25, (0,255,0), 1)
                    img = cv.rectangle(img, (x1,y1), (x2,y2), color=(0,255,0), thickness=1)
                detected_boxes = []
                for b, s, c in zip(boxes, scores, labels):
                    if s>0.9: 
                        x1, y1, x2, y2 = b
                        img = cv.rectangle(img, (x1,y1), (x2,y2), color=(0,0,255), thickness=1)
                        y = y1 - 9 if y1 - 9 > 9 else y1 + 9
                        cv.putText(img, classes_by_num[c], (x1, y), cv.FONT_HERSHEY_SIMPLEX, 0.25)

                cv.imwrite(f"{burn_on_dir}/img_{i}.png", img)
        loss_hist.send_val_iou_precision(
                                            np.mean(val_image_ious), 
                                            np.mean(val_image_precisions)
                                        )
        
       

    loss_hist.log_val

def fit(
        model, 
        train_data_loader, 
        valid_data_loader, 
        num_epochs, 
        optimizer, 
        device,
        loss_hist,
        lr_scheduler=None,
        checkpoint_dir=None
        ):
    for epoch in range(num_epochs):
        start_time = time()
        _train_one_epoch(model, train_data_loader, optimizer, device, loss_hist)
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        models = [model]
        evaluate(models, valid_data_loader, device, loss_hist)

        logger.info(f"Epoch #{epoch+1} "
                    f"Train loss: {loss_hist.trn_value:.5f} "
                    f"Test Accuracy: {loss_hist.iou_value:.5f} "
                    f"Time taken : {str(datetime.timedelta(seconds=time() - start_time))[:7]}"
                )

        loss_hist.plot_losses
        loss_hist.epoch += 1    

        is_improved = loss_hist.set_best_val
        if is_improved=="improved":
            n = datetime.datetime.now().strftime("%M-%H-%d-%m-%y")
            logger.info(f"==> Saving model to {checkpoint_dir}/model_{n}.pth ...")
            torch.save(model, f"{checkpoint_dir}/chkpnt_{n}.pth")  # Saving current best model torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
        elif is_improved=="Quit":
            break
        loss_hist.reset