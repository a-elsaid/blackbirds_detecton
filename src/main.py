import torch
from split_data import DataPrep
from data import setup_data_loaders
from os import makedirs
from loguru import logger
from augmentation import get_train_transform
from helper import LossHistory, DEVICE
from model import CustomModel
from engine import fit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir', type=str, required=True)
parser.add_argument('-lg', '--log_dir', type=str, required=True)
parser.add_argument('-mn', '--model_name', type=str, required=False)
parser.add_argument('-d', '--data_path', type=str, required=True)
parser.add_argument('-t', '--tiles_dir', default="plain_tiles", type=str, required=False)
parser.add_argument('-l', '--labels_path', type=str, required=True)
parser.add_argument('-e', '--num_epochs', type=int, default=100000)
parser.add_argument('-s', '--data_shuffle', action='store_true')
parser.add_argument('-c', '--use_4chn', action='store_true')
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-f', '--read_from_files', action='store_true')
parser.add_argument('-lrs', '--lr_scheduler', action='store_true')
parser.add_argument('-ut', '--use_tiles', action='store_true')
parser.add_argument('-ct', '--create_tiles', action='store_true')


args = parser.parse_args()
use_4chn = args.use_4chn
model_path = args.model_dir
tiles_folder = args.tiles_dir
log_dir = args.log_dir
model_name = args.model_name 
data_path = args.data_path
label_path= args.labels_path
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_sched = args.lr_scheduler
use_tiles = args.use_tiles
create_tiles = args.create_tiles
read_from_files = args.read_from_files


logger.info(f"Model Directory: {model_path}")
logger.info(f"Log Directory: {log_dir}")
logger.info(f"Tiles Directory: {tiles_folder}")
logger.info(f"Data Path: {data_path}")
logger.info(f"Labels Path: {label_path}")   
logger.info(f"Number of Epochs: {num_epochs}")
logger.info(f"Batch Size: {batch_size}")
logger.info(f"Read from Files: {args.read_from_files}")
logger.info(f"Use Tiles: {use_tiles}")
logger.info(f"Create Tiles: {create_tiles}")
logger.info(f"Read from Files: {read_from_files}")

makedirs(tiles_folder, exist_ok=True)
makedirs(model_path, exist_ok=True)
makedirs(log_dir, exist_ok=True)

trans_fun = get_train_transform

data = DataPrep(
                img_dir=data_path,
                label_dir=label_path,
                d_x= 180,
                d_y= 240,
                tiling=use_tiles, 
                shuffle = args.data_shuffle,
                read_from_files = read_from_files,
                create_tiles = create_tiles,
                save_to_tiles_dir=tiles_folder
                )

logger.info(f"Number of Classes: {data.num_classes}")
logger.info(f"Classes: {data.classes}")
logger.info(f"Training Data Size: {len(data.train_test_sets['train'])}")
logger.info(f"Validation Data Size: {len(data.train_test_sets['test'])}")


trn_val_data = setup_data_loaders(batch_size, tiles_folder,use_4chn, data)
train_data_loader, valid_data_loader = trn_val_data

model = CustomModel(num_classes=data.num_classes, classes=data.classes).to(DEVICE)
logger.info("Model Structure..."); #print(model)
models = [model]

optimizer = torch.optim.SGD(model.num_grad_params, lr=0.0001, momentum=0.87, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) if lr_sched else None

loss_hist = LossHistory(log_dir=log_dir)
fit(
    model, 
    train_data_loader, 
    valid_data_loader, 
    num_epochs, 
    optimizer, 
    DEVICE, 
    loss_hist, 
    lr_scheduler,
    model_path,
    )