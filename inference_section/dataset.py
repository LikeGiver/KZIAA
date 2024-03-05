import torch
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from models.coop import *
from clip_code import clip
from utils.base_utils import *
import pickle

class MyDataset(Dataset):
    def __init__(self, device, text_features):
        self.text_features = text_features

        # get the classname_indexes
        with open('data/class_names.pkl', 'rb') as f:
            class_names_list = pickle.load(f)
        classname_indexes = []
        classnames = ['General Impression',  # just use the 7 classes string list 
            'Subject of Photo',
            'Composition & Perspective',
            'Use of Camera,Exposure & Speed',
            'Depth of Field',   
            'Color & Lighting',
            'Focus']
        for class_name in class_names_list:
            classname_indexes.append(classnames.index(class_name))
        self.classname_indexes = classname_indexes
        ############################

        #get train_photo_ids
        with open('data/train_photo_ids.pkl', 'rb') as f:
            train_photo_ids = pickle.load(f)
        self.train_photo_ids = train_photo_ids
        ######################

        #get table_photo_ids
        with open('data/table_photo_ids.pkl', 'rb') as f:
            table_photo_ids = pickle.load(f)
        self.table_photo_ids = table_photo_ids
        ######################

        # load features from the pt file
        self.device = device
        self.image_features = torch.load('data/image_features.pt').to(self.device)
        
        self.labels = torch.load('data/sentence_tensor.pt').to(self.device)


    def __getitem__(self, idx):
        photo_id = self.train_photo_ids[idx]
        table_idx = self.table_photo_ids.index(photo_id)
        preprocessed_image_tensor = self.image_features[table_idx]
        text_index = self.classname_indexes[idx]
        concat_feature = torch.stack([preprocessed_image_tensor,self.text_features[text_index]], dim = 0).type(torch.float32)

        # 处理文本 changed idx to 0
        label = self.labels[idx]

        # 返回样本及其标签
        return concat_feature, label
    

    def __len__(self):
        return len(self.labels)

