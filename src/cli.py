import torch
import argparse, json
from pathlib import Path
from loguru import logger
from data import setup_data_loaders
from model import CustomModel
from split_data import DataPrep
from helper import DEVICE, LossHistory
from config import Config, TrainConfig, DetectionConfig, load_config
from engine import fit, evaluate

def _build_datasets(cfg: Config):
    data = DataPrep(
                img_dir=cfg.data_path,
                label_dir=cfg.labels_path,
                d_x=cfg.d_x,
                d_y=cfg.d_y,
                tiling=cfg.use_tiles,
                shuffle=cfg.data_shuffle,
                read_from_files=cfg.read_from_files,
                create_tiles=cfg.create_tiles,
                save_to_tiles_dir=cfg.tiles_dir
                )

    logger.info(f"Number of Classes: {data.num_classes}")
    logger.info(f"Classes: {data.classes}")
    logger.info(f"Training Data Size: {len(data.train_test_sets['train'])}")
    logger.info(f"Validation Data Size: {len(data.train_test_sets['test'])}")
  

    return setup_data_loaders(cfg.batch_size, cfg.tiles_dir, cfg.use_4chn, data), data.num_classes, data.classes

def cmd_train(args):
    logger.info("Starting Training -- Loading configuration...")
    raw = load_config(args.config)

    cfg = TrainConfig(**raw["train"])
    (train_data_loader, valid_data_loader), num_classes, classes = _build_datasets(cfg)

    model = CustomModel(num_classes=num_classes, classes=classes).to(DEVICE)
    logger.info("Model Structure..."); #print(model)
    models = [model]

    optimizer = torch.optim.SGD(model.num_grad_params, lr=0.0001, momentum=0.87, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) if cfg.lr_scheduler else None

    loss_hist = LossHistory(log_dir=cfg.log_dir)
    fit(
        model, 
        train_data_loader, 
        valid_data_loader, 
        cfg.num_epochs, 
        optimizer, 
        DEVICE, 
        loss_hist, 
        lr_scheduler,
        cfg.model_dir,
        )
    logger.info("Training complete.")

def cmd_predict(args):
    raw = load_config(args.config)
    cfg = DetectionConfig(**raw["detect"])
    (_, valid_data_loader), num_classes, classes = _build_datasets(cfg)

    logger.info("Model Structure..."); #print(model)
    models = [torch.load(name).to(DEVICE) for name in cfg.models_names]

    loss_hist = LossHistory(log_dir=cfg.log_dir)
    evaluate(
        models=models,
        data_loader=valid_data_loader,
        device=DEVICE,
        loss_hist=loss_hist,
        burn_on_dir=cfg.burn_results
    )

def build_parser():
    ap = argparse.ArgumentParser(description="Black Bird Detection")
    sp = ap.add_subparsers(dest="cmd")

    p_train   = sp.add_parser("train", help="Train model")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=cmd_train)

    p_predict = sp.add_parser("detect", help="Use trained model to detect birds")
    p_predict.add_argument("--config", required=True)
    p_predict.set_defaults(func=cmd_predict)

    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help(); return
    args.func(args)

if __name__ == "__main__":
    main()