import cv2 as cv
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from os.path import join, splitext, basename

def read_labels(path, name):
    d_name = os.path.join(os.path.dirname(name), path)
    name = splitext(basename(name))[0]+'.xml'
    print(os.path.join(d_name,name))
    tree = ET.parse(os.path.join(d_name,name))
    root = tree.getroot()
    labels = []
    print(root[1].text)
    for i in root.findall("object"):
        labels.append([
                        os.path.basename(name).replace('.xml',''),
                        i[0].text,
                        int(i[4][0].text),
                        int(i[4][1].text),
                        int(i[4][2].text),
                        int(i[4][3].text),
                       ])
        print ("Label:", labels[-1])
    return labels


def tile(labels_file, name, d_x, d_y, x_pad=50, y_pad=50):
    fl = open(labels_file, 'a')
    img = cv.imread(name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    raw_labels = read_labels("TrainingAnnotations", name)
    '''
    for l in raw_labels:
        n, lx1, ly1, lx2, ly2 = l
        cv.imwrite("org_img.jpg",  cv.rectangle(img, (lx1,ly1), (lx2,ly2), (255,0,0), 2))
    '''
    
    for m, i in enumerate(range(0,img.shape[0], d_x)):
        for n, j in enumerate(range(0,img.shape[1], d_y)):
            s_x = i-x_pad
            s_y = j-y_pad
            e_x = i+d_x
            e_y = j+d_y
            if i==0: 
                s_x = i
                e_x = e_x + x_pad
            if j==0: 
                s_y = j
                e_y = e_y + y_pad

            im =  img[s_x:e_x, s_y:e_y]   
            allowx = 0.10*d_x
            allowy = 0.10*d_y
            for l in raw_labels:
                img_id, n, ly1, lx1, ly2, lx2 = l
                if lx1+allowx>=s_x and ly1+allowy>=s_y and lx2-allowx<=e_x and ly2-allowy<=e_y:
                    w = ly2-ly1
                    h = lx2-lx1
                    label = f'{img_id},{im.shape[1]},{im.shape[0]},{ly1},{lx1},{w},{h},{n},{w*h}\n'
                    fl.write(label)
                    lx1-=s_x
                    lx2-=s_x
                    ly1-=s_y
                    ly2-=s_y
                    im = cv.rectangle(im, (ly1,lx1), (ly2,lx2), (255,0,0), 2)
                    print(label)

            cv.imwrite(os.path.basename(name).replace(" ", "").replace(".jpg", "") + f"_{i}-{i+d_x}_{j}-{j+d_y}" + ".jpg", im)
            
            '''
            cv.imshow(f"image_{i}-{i+d_x}_{j}-{j+d_y}", im)

            key = cv.waitKey()
            if cv.waitKey() & 0xFF==ord(' ') : 
                cv.destroyAllWindows()
            else:
                exit()
            '''
    fl.close()

labels_file = "labels.csv"
with open(labels_file, 'w') as fl:
    fl.write("img_id,width,height,x,y,w,h,class,area\n")
img_dir= "../Training Dataset"
for f in (
                glob(join(img_dir,"*.jpg")) + 
                glob(join(img_dir,"*.JPG")) + 
                glob(join(img_dir,"*.png"))
                ):
    #tile(img, 240, 240)
    tile(labels_file, f, 270, 320)
    #break
