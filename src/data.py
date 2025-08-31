from torch.utils.data import DataLoader
from data_loader import DataSet
from helper import collate_fn


def setup_data_loaders(
                        batch_size,
                        tiles_folder, 
                        use_4chn,
                        data,
    ):


    train_data = DataSet(
                    data=data, 
                    image_dir=data.img_dir,
                    # transforms=trans_fun(),
                    transforms=None,
                    mode='train',
                    tiles_folder=tiles_folder,
                    use_4chn = use_4chn,
                )

    test_data = DataSet(
                    data=data, 
                    image_dir=data.img_dir,
                    # transforms=trans_fun(),
                    transforms=None,
                    mode='test',
                    tiles_folder=tiles_folder,
                    use_4chn = use_4chn,
                )

    train_data_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    valid_data_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return train_data_loader, valid_data_loader