import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, X_images, labels, train_photo_ids, table_photo_ids):
        self.X_images = X_images
        self.labels = labels
        self.train_photo_ids = train_photo_ids
        self.table_photo_ids = table_photo_ids

    def __getitem__(self, idx):
        # 读取图片
        # img = self.X_images[idx]
        photo_id = self.train_photo_ids[idx]
        table_idx = self.table_photo_ids.index(photo_id)
        img = self.X_images[table_idx]

        # 处理文本
        label = self.labels[idx]

        # 返回样本及其标签
        return img, label

    def __len__(self):
        return len(self.labels)

