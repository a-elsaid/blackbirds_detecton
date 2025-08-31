import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
# from os.path import basename, join, dirname, splitext
from glob import glob
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import cv2 as cv
from bg_remove_histogram import remove_dominant_colors
from loguru import logger
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}

@dataclass
class DataPrep:
    img_dir: Path | str
    label_dir: Path | str | None = None
    splits_names_dir: Path | str | None = '.'
    tiling: bool = False
    train_test_ratio: float = 0.75
    create_tiles: bool = False
    d_x: Optional[int] = None
    d_y: Optional[int] = None
    model_type: Optional[str] = None
    shuffle: bool = False
    read_from_files: bool = True
    seed: Optional[int] = 42
    save_to_tiles_dir: Path | str = "."

    # internal fields
    classes: Dict[str, int] = field(default_factory=dict, init=False)
    labels: Dict[str, List[List[int | str | float]]] = field(default_factory=dict, init=False)
    num_classes: int = field(default=0, init=False)
    train_test_sets: Dict[str, List[str]] = field(default_factory=lambda: {"train": [], "test": []}, init=False)
    frames_names: List[str] = field(default_factory=list, init=False)


    def __post_init__(self) -> None:
        self.img_dir = Path(self.img_dir)
        self.label_dir = Path(self.label_dir) if self.label_dir else None
        self.save_to_tiles_dir = Path(self.save_to_tiles_dir)
        self.splits_names_dir = Path(self.splits_names_dir)

        self.splits_names_dir.mkdir(parents=True, exist_ok=True)
        self.save_to_tiles_dir.mkdir(parents=True, exist_ok=True)

        self.labels = {}

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if self.label_dir and not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        np.random.seed(self.seed)

        logger.info(f"Scanning images in  {self.img_dir}...")

        files_names =   [
                                f.stem for f in self.img_dir.glob("Trial*") 
                                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]#[:1] # ****** REMOVE REMOVE REMOVE REMOVE REMOVE REMOVE ******  

        self.imgs_names = []
        for img_name in tqdm(files_names, total=len(files_names), desc="Processing images", leave=False):
            logger.info(f"Processing image: {img_name}")
            img_raw_labels = self.__read_xml_labels(self.label_dir, img_name) if self.label_dir else []
            if self.tiling:
                self.__create_tiles(raw_labels=img_raw_labels, d_x=self.d_x, d_y=self.d_y, img_name=img_name)
                self.frames_names.append(img_name)
            else:
                self.labels[img_name] = img_raw_labels


        if self.shuffle:
            logger.info("Shuffling data...")
            np.random.shuffle(self.frames_names)


        
        logger.info("Splitting data into train and test sets...")
        train_test_limit = int(len(self.imgs_names) * self.train_test_ratio)
        self.train_test_sets['train'] = self.imgs_names[:train_test_limit]
        self.train_test_sets['test'] = self.imgs_names[train_test_limit:]


        def file_read(file_path: str) -> List[str]:
            with open(file_path, 'r') as f:
                return f.readlines()

        def file_write(file_path: str, files_names: Dict[str, List[str]], set_name: str) -> None:
            with open(file_path, 'w') as f:
                for n in files_names[set_name]: f.write(f"TILE{n}\n")

        if self.read_from_files:
            logger.info("Reading train and test set file names from disk...")
            self.train_test_sets['train'] = file_read('train_set_files.txt')
            self.train_test_sets['test'] = file_read('test_set_files.txt')
        else:
            logger.info("Saving training and test set file names to disk...")
            file_write("train_set_files.txt", self.train_test_sets, set_name='train')
            file_write("test_set_files.txt",  self.train_test_sets, set_name='test')

    def __read_xml_labels(self, path, img_name) -> List:
        xml_path = Path(path) / f"{img_name}.xml"
        logger.info(f"Reading labels from {xml_path}")
        if not xml_path.exists():
            logger.warning(f"Label file not found: {xml_path}, skipping labels for this image.")
            return []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.warning(f"Error parsing XML file {xml_path}: {e}")
            return []
        
        size_elem = root.find("size")
        if size_elem is None:
            logger.warning(f"No size element found in XML file {xml_path}, skipping labels for this image.")
            return []


        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)

        labels = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            x1 = int(bbox.find("xmin").text)
            y1 = int(bbox.find("ymin").text)
            x2 = int(bbox.find("xmax").text)
            y2 = int(bbox.find("ymax").text)
            w = x2 - x1
            h = y2 - y1
            if w==0 or h==0: continue
            area = w*h
            (u,v) = (w, h) if self.model_type == "retinanet" else (x2, y2)
            labels.append([img_name, class_name, width, height, x1, y1, u, v, area])
            
            if class_name not in self.classes:
                self.classes[class_name] = len(self.classes) + 1
                self.num_classes+=1
        return labels

    def __create_tiles(
                        self,
                        raw_labels: List[List[int | str | float]],
                        d_x: int,
                        d_y: int,
                        x_pad: int = 50,
                        y_pad: int = 50,
                        img_name: Optional[str] = None,
    ) -> None:
        assert raw_labels and d_x and d_y, "raw_labels, d_x and d_y are required"
        im_w = raw_labels[0][2] # width of the image is logged in all the labels at idx 2
        im_h = raw_labels[0][3] # height of the image is logged in all the labels at idx 3
        im_w = (im_w // d_y) * d_y
        im_h = (im_h // d_x) * d_x
        allow_x, allow_y = np.array([d_x, d_y]) * 0.10 # allowance in x, y 
        
    
        if self.create_tiles:
            for ext in IMAGE_EXTS:
                img_path = self.img_dir / f"{img_name}{ext}"
                img = cv.imread(img_path.as_posix(), cv.IMREAD_UNCHANGED)
                if img is not None: break
            img = remove_dominant_colors(img)

        for m, i in enumerate(range(0, im_h, d_x)):
            for n, j in enumerate(range(0, im_w, d_y)):
                s_x = max(i - x_pad, 0)
                s_y = max(j - y_pad, 0)
                e_x = min(i + d_x, im_h) if i!=0 else min(i + d_x + x_pad, im_h)
                e_y = min(j + d_y, im_w) if j!=0 else min(j + d_y + y_pad, im_w)

                image_id = f"{img_name}_{i}_{i+d_x}_{j}_{j+d_y}"
                tile_img_name = f"TILE{image_id}"
                if self.create_tiles:
                    tile = img[s_x:e_x, s_y:e_y]
                    cv.imwrite((self.save_to_tiles_dir/f"{tile_img_name}.png").as_posix(), tile)

                for l in raw_labels:
                    img_id, class_name, _, _, x1, y1, x2, y2, area = l
                    assert img_id == img_name, f"Image ID mismatch: {img_id} vs {img_name}"
                    if (
                        y1 + allow_y > s_x and  
                        x1 + allow_x > s_y and 
                        y2 - allow_y < e_x and 
                        x2 - allow_x < e_y
                    ):
                        y1 -= s_x
                        x1 -= s_y
                        y2 -= s_x
                        x2 -= s_y

                        h = e_x - s_x
                        w = e_y - s_y

                        label = [
                                 img_id, class_name,
                                 w, h, 
                                 x1, y1, x2, y2,
                                 area,
                                 s_y,s_x, e_y, e_x,
                                 img_name, "TILE",
                                ]

                        if self.create_tiles:
                            label_file_path = self.save_to_tiles_dir / f"{tile_img_name}.txt"
                            with open(label_file_path, 'a') as label_file:
                                label_file.write(f"{self.classes[class_name]} {x1/w} {y1/h} {(x2-x1)/w} {(y2-y1)/h}\n")

                        if tile_img_name not in self.imgs_names:
                            self.imgs_names.append(tile_img_name)

                        self.labels.setdefault(img_name, []).append(label)


    # def __init__(
    #             self, 
    #             img_dir, 
    #             label_dir=None, 
    #             tiling=False, 
    #             train_test_ratio=0.75, 
    #             create_tiles=False, 
    #             d_x=None, 
    #             d_y=None,
    #             model_type=None,
    #             shuffle = False,
    #             read_from_files = True,
    #             ):
    #     self.img_dir = img_dir
    #     self.label_dir = label_dir
    #     self.create_tiles = create_tiles
    #     self.tiling = tiling
    #     self.model_type = model_type

    #     self.classes = {}
    #     self.labels = {}
    #     self.num_classes = 0
    #     self.train_test_sets = {'train': [], 'test':[]}

        # if not self.label_dir:
        #     self.label_dir = self.img_dir

        # self.imgs_names = []



    #     for img_name in (
    #                     glob(join(img_dir,"*.jpg")) + 
    #                     glob(join(img_dir,"*.JPG")) + 
    #                     glob(join(img_dir,"*.png"))
    #                     ):
    #         im_name = basename(img_name)
    #         raw_labels = self.read_xml_labels(label_dir, im_name)
    #         im_name = splitext(im_name)[0]
    #         if self.tiling:
    #             self.tile(raw_labels, d_x=d_x, d_y=d_y , img_name=img_name)
    #         else:
    #             self.labels[im_name] = raw_labels
    #             self.imgs_names.append(splitext(basename(im_name))[0])
        
    #     if shuffle:
    #         print ("Shuffling Data...")
    #         np.random.shuffle(self.imgs_names)
    #     train_test_limit = int(len(self.imgs_names) * train_test_ratio)
    #     self.train_test_sets['train'] = self.imgs_names[:train_test_limit]
    #     self.train_test_sets['test'] = self.imgs_names[train_test_limit:]

    #     """
    #     """
    #     if read_from_files:
    #         with open('train_set_files.txt', 'r') as f:
    #             self.train_test_sets['train'] = f.readlines()
    #         with open('test_set_files.txt', 'r') as f:
    #             self.train_test_sets['test'] = f.readlines()
    #     else:
    #         print("Saving Traing ang Testing Tile Names")
    #         with open("train_set_files.txt", 'w') as f:
    #             for n in self.train_test_sets['train']:
    #                 f.write(f"TILE{n}\n")
    #         with open("test_set_files.txt", 'w') as f:
    #             for n in self.train_test_sets['test']:
    #                 f.write(f"TILE{n}\n")

    #     self.read_from_files = read_from_files    

    # def read_xml_labels(self, path, name) -> List:
    #     d_name = join(dirname(name), path)
    #     name = splitext(basename(name))[0]
    #     tree = ET.parse(join(d_name,name+".xml"))
    #     root = tree.getroot()
    #     labels = []
    #     img_size = root[4]
    #     width = int(img_size[0].text)
    #     height = int(img_size[1].text)
    #     for i in root.findall("object"):
    #         # im_id, class, width, height, x, y, w, h, area
    #         x1 = int(i[4][0].text)
    #         x2 = int(i[4][2].text)
    #         y1 = int(i[4][1].text)
    #         y2 = int(i[4][3].text)
    #         w = x2 - x1
    #         h = y2 - y1
    #         if w==0 or h==0:
    #             continue
    #         a = w*h
    #         Z1 = x2
    #         Z2 = y2
    #         if self.model_type=="retinanet":
    #             Z1 = w
    #             Z2 = h
    #         '''
    #         '''
    #         cls = i[0].text
    #         # cls = "bird"
    #         labels.append([
    #                         name,
    #                         cls,
    #                         width,
    #                         height,
    #                         x1,
    #                         y1,
    #                         Z1,
    #                         Z2,
    #                         a,
    #                        ])
    #         if cls not in self.classes:
    #             if len(self.classes) == 0:
    #                 self.classes[cls] = 1
    #             else:
    #                 self.classes[cls] = max(self.classes.values()) + 1
    #             self.num_classes+=1
    #     return labels
       

    # def tile(self, raw_labels, d_x, d_y, x_pad=50, y_pad=50, img_name=None) -> None:
    #     labels = []
    #     img_width = raw_labels[0][2]
    #     img_height = raw_labels[0][3]

    #     img_height = (img_height//d_x) * d_x
    #     img_width = (img_width//d_y) * d_y
        
    #     for m, i in enumerate(range(0,img_height, d_x)):
    #         for n, j in enumerate(range(0,img_width, d_y)):
    #             s_x = i-x_pad
    #             s_y = j-y_pad
    #             e_x = i+d_x
    #             e_y = j+d_y
    #             if i==0: 
    #                 s_x = i
    #                 e_x = e_x + x_pad
    #             if j==0: 
    #                 s_y = j
    #                 e_y = e_y + y_pad

    #             allowx = 0.0 #0.10*d_x
    #             allowy = 0.0 #0.10*d_y

    #             if self.create_tiles:
    #                 img = cv.imread(
    #                                 img_name.replace(".jpg", ".png"), 
    #                                 # img_name,
    #                                 cv.IMREAD_UNCHANGED,
    #                                )
    #                 #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #                 im =  img[s_x:e_x, s_y:e_y]   


    #             if self.create_tiles:
    #                 label_file = open("../plain_tiles/TILE" + basename(img_name).replace(".jpg", "") + f"_{i}_{i+d_x}_{j}_{j+d_y}" + ".txt", 'a')
    #             for l in raw_labels:
    #                 img_id, n, _, _, x1, y1, x2, y2, a = l
    #                 if (y1+allowx>s_x and  
    #                     x1+allowy>s_y and 
    #                     y2-allowx<e_x and 
    #                     x2-allowy<e_y):

    #                     y1-=s_x
    #                     x1-=s_y
    #                     y2-=s_x
    #                     x2-=s_y

    #                     h = e_x - s_x
    #                     w = e_y - s_y

    #                     label = [
    #                              img_id,
    #                              n,
    #                              w, 
    #                              h, 
    #                              x1, 
    #                              y1, 
    #                              x2, 
    #                              y2,
    #                              a,
    #                              s_y,
    #                              s_x,
    #                              e_y,
    #                              e_x,
    #                              basename(img_name),
    #                              "TILE",
    #                             ]

    #                     if self.create_tiles:
    #                         label_file.write(f"{self.classes[n]} {x1/w} {y1/h} {(x2-x1)/w} {(y2-y1)/h}\n")
    #                         """
    #                         label_file.write(f"{1} {x1/w} {y1/h} {(x2-x1)/w} {(y2-y1)/h}\n")
    #                         """

    #                     if splitext(basename(img_name))[0] + f"_{i}_{i+d_x}_{j}_{j+d_y}" not in self.imgs_names:
    #                         self.imgs_names.append(splitext(basename(img_name))[0] + f"_{i}_{i+d_x}_{j}_{j+d_y}")

    #                     im_name = img_id + f"_{i}_{i+d_x}_{j}_{j+d_y}"
    #                     if im_name in self.labels:
    #                         self.labels[im_name].append(label)
    #                     else:
    #                         self.labels[im_name] = [label]

    #                     #if self.create_tiles: im = cv.rectangle(im, (x1,y1), (x2,y2), (255,0,0), 2)


    #             if self.create_tiles: 
    #                 cv.imwrite("../plain_tiles/TILE" + basename(img_name).replace(".jpg", "") + f"_{i}_{i+d_x}_{j}_{j+d_y}" + ".png", im)
    #                 label_file.close()
       

