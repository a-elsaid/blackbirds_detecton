import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import isfile
from loguru import logger



def load_image(file_path):
    im = cv.imread(file_path, cv.IMREAD_UNCHANGED).astype(np.float32)
    if im is None:
        logger.error(f"File {file_path} DOES NOT EXIST")
        exit()
    return im


class DataSet(Dataset):

    def __init__(
            self, 
            data, 
            image_dir, 
            transforms=None, 
            mode='train', 
            tiles_folder="plain_tiles",
            use_4chn=True,
        ):
        super().__init__()

        self.im_ids = data.train_test_sets[mode]
        self.image_dir = image_dir
        self.transforms = transforms
        self.frame_name = None
        self.image = None
        self.data = data
        self.verbose = False
        self.use_4chn = use_4chn
        self.tiles_folder = tiles_folder 
        self.im = None

    def __get_img__(self, index: int):
        img_path = self.tiles_folder / (self.im_ids[index][4:].strip() + ".png")
        img = cv.imread(str(img_path))
        return img  

    def __getitem__(self, index: int):
        #records = self.df[sdd`elf.df['im_id'] == im_id] #Getting all coordinates for the given image
        im_id = self.im_ids[index].strip()
        img_id = im_id.replace('TILE','').strip('.jpg')
        image_id = '_'.join(img_id.replace('TILE','').split('_')[:-4])
        records = self.data.labels[image_id]

        if self.data.tiling:        # if using tiles as input
            im_ext = 'png' if self.use_4chn else 'jpg'
            im_id, cls, w, h, _, _, _, _, _, s_y, s_x, e_y, e_x, self.frame_name = records[0][:-1]
            img_id = "TILE" + img_id
            if self.data.read_from_files:       # if loading saved tiles
                im = load_image(f"{self.tiles_folder}/{img_id}.{im_ext}")
            else:                               # if using tiles on the fly
                if self.frame_name != image_id or self.im is None:
                    self.frame_name = image_id
                    self.image = load_image(f'{self.image_dir}/{self.frame_name}.jpg')
                im = self.image[s_x:e_x, s_y:e_y]
            im /= 255.0
        else:                   # if using full images
            self.frame_name = im_id
            image = load_image(f'{self.image_dir}/{im_id}.jpg')
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            im = image

        boxes = []
        area = []
        labels = []
        for r in records: 
            b = r[4:8]
            if np.any(np.array(b)<0)  or b[0]-b[2]==0 or b[1]-b[3]==0:
                continue
            boxes.append(b)
            if self.verbose:
                logger.info(boxes[-1])
            area.append(r[8])
            labels.append(self.data.classes[r[1]])

        if len(boxes)==0:
            boxes.append([0,0,1,1])
            area.append(1)
            labels = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # there is only one class
            # labels = [x[1] for x in records]
            ...
            # labels = torch.ones((len(boxes),), dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int32)

        boxes = torch.as_tensor(boxes,dtype=torch.int64)
        area = torch.as_tensor(area,dtype=torch.int32)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['im_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.verbose:
            logger.info('BOXES:',boxes)
            logger.info('IMAGE NAME:', self.frame_name)
            #logger.info(f'Sx: {s_x}, Ex: {e_x} \n S_y: {s_y}, E_y: {e_y}')
            logger.info('-'*40)

        if self.transforms:
            if self.verbose: logger.info("AUGMENTING IMAGE")
            sample = {
                'image': im,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            im  = sample['image'] #.type(torch.uint8)
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        else:
            if self.verbose: logger.info("**NOT** AUGMENTING IMAGE")
            im = im.transpose((2, 0, 1))
            im = torch.as_tensor(im, dtype=torch.float32)
        
        return im, target, im_id
        # return target

    def __len__(self) -> int:
        return len(self.im_ids)


