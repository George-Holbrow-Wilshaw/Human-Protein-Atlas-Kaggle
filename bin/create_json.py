import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
import pandas as pd
import cv2
import os


def image_entry(fn, h, w, id_):
    fp = '../input/hpa-single-cell-image-classification/test/'
    print(fp + fn)
    im = cv2.imread(fp + fn)
    images = {
        "file_name": fn,
        "height": im.shape[0],
        "width": im.shape[1],
        "id": id_
    }
    return images

def cat_entry(c, id_):
    cat = {"supercategory": c,"id": int(id_),"name": c}
    return cat

def annotation_entry(seg, im_id, bbox, c_id, id_):
    annotation = {
        "segmentation": seg,
        "area": 728,
        "iscrowd": 0,
        "image_id": im_id,
        "bbox": bbox,
        "category_id": c_id,
        "id": id_
    }
    return annotation

def convert_df_to_JSON(df, cats):
    fp = '../input/hpa-single-cell-image-classification/test/'
    ann_con = []
    cat_con = []
    im_con = []
    count = 0
    for c in cats:
        cat_con.append(cat_entry(c, c)) 
    for r in df.index:
        count += 1
        row = df.iloc[r,:]
        id_ = row['ID']
        cats = list(row['Label'].split('|'))
        i = image_entry(id_ + '_green.png', 728, 728, id_)
        im_con.append(i)
        bbox = [50, 50, 20, 20]
        for c in cats:
            count += 1
            seg = 0
            a = annotation_entry(seg, id_, bbox, int(c), count)
            ann_con.append(a)
    obj = {'annotations': ann_con, 'categories': cat_con, 'images': im_con}
    return obj
