import numpy as np
import pandas as pd
import cv2 
from tqdm import tqdm
import pickle
import hpa_config as cfg

def extract_bboxes(masks):
    bboxes = []
    n_masks = np.unique(masks)
    for i in n_masks[1:]:
        mask = masks.copy()
        mask[mask != i] = 0
        mask = np.array(mask, dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(mask)
        bboxes.append(np.array([x, y, w, h]))
    return np.array(bboxes)


def get_all_bboxes(fns):
    
    all_ = []
    
    for fn in tqdm(fns):
        cell_mask = np.load('masks/hpa_cell_mask/' + fn + '.npz')['arr_0']
        all_.append(extract_bboxes(cell_mask))
        
    with open(f'data/bboxes_{cfg.DATASET}.pkl', 'wb') as f:
        pickle.dump(all_, f)
            
    return all_


def create_train_file(bboxes, train_df):
    scores = [np.ones(b.shape[0]) for b in bboxes]
    indexes = train_df['ID']
    tf = {'boxes':bboxes, 'indexes':indexes, 'scores':scores}
    
    with open('/data/HPA_train.pkl', 'wb') as f:
        pickle.dump(tf, f)
        
    return tf