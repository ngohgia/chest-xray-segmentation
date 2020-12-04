from skimage.transform import resize
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv

import numpy as np
import pydicom

import PIL

from utilities import *

def load_data(train_csv_path, test_csv_path, train_images_dir, test_images_dir, batch_size, validation_prop, rescale_factor):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    train_df = train_df.sample(frac=1, random_state=42)
    pids = [pid for pid in train_df['patientId'].unique()]
    
    dev_pids = pids[ : int(round(validation_prop * len(pids)))]
    train_pids = pids[int(round(validation_prop * len(pids))) : ]
    test_pids = test_df['patientId'].unique()

    train_df['box_area'] = train_df['width'] * train_df['height']
    min_box_area = train_df['box_area'].min()
    
    boxes_by_pid_dict = {}
    for pid in train_df.loc[(train_df['Target'] == 1)]['patientId'].unique().tolist():
        boxes_by_pid_dict[pid] = get_patient_boxes_values(train_df, pid)

    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    
    train_dataset = PneumoniaDataset(images_data_dir=train_images_dir, pids=train_pids, is_predict=False,
                                     boxes_by_pid_dict=boxes_by_pid_dict, rescale_factor=rescale_factor, transform=transform,
                                    rotation_angle=3, warping=True)
    dev_dataset = PneumoniaDataset(images_data_dir=train_images_dir, pids=dev_pids, is_predict=False,
                                     boxes_by_pid_dict=boxes_by_pid_dict, rescale_factor=rescale_factor, transform=transform,
                                    rotation_angle=0, warping=False)
    dev_dataset_for_predict = PneumoniaDataset(images_data_dir=train_images_dir, pids=dev_pids, is_predict=True,
                                     boxes_by_pid_dict=None, rescale_factor=rescale_factor, transform=transform)
    test_dataset = PneumoniaDataset(images_data_dir=test_images_dir, pids=test_pids, is_predict=True,
                                     boxes_by_pid_dict=None, rescale_factor=rescale_factor, transform=transform,
                                    rotation_angle=0, warping=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)
    dev_loader_for_predict = DataLoader(dataset=dev_dataset_for_predict, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_df, train_loader, dev_pids, dev_loader, dev_dataset_for_predict, dev_loader_for_predict, test_loader, test_df, test_pids, boxes_by_pid_dict, min_box_area


class PneumoniaDataset(torchDataset):
    def __init__(self, images_data_dir, pids, is_predict, boxes_by_pid_dict, rescale_factor=1, transform=None, rotation_angle=0, warping=False):
        """
            images_data_dir: path to directory containing the images
            pids: lsit of patient IDs
            is_predict: if true, returns iamge and target labels, otherwise return images
            boxes_by_pid_dict: dictionary of the format { patientId: list of bounding boxes }
            rescale_factor: image rescale factor
            transform: transformation applied to images and target masks
            rotation_angle: float number defining range of rotation angles for augmentation (-rotation_angle, +rotation_angle)
            warping: boolean, if true applying augmentation warping to image, do nothing otherwise
        """

        self.images_data_dir = os.path.expanduser(images_data_dir)
        self.pids = pids
        self.is_predict = is_predict
        self.boxes_by_pid_dict = boxes_by_pid_dict
        self.rescale_factor = rescale_factor
        self.transform = transform
        self.rotation_angle = rotation_angle
        self.warping = warping

        self.images_path = images_data_dir

    def __getitem__(self, index):
        """
            index: index of the pid
        """
        pid = self.pids[index]

        img = pydicom.dcmread(os.path.join(self.images_path, pid + '.dcm')).pixel_array
        original_image_dim = img.shape[0]
        image_dim = int(original_image_dim / self.rescale_factor)

        img = resize(img, (image_dim, image_dim), mode='reflect')
        img = min_max_scale_image(img, (0, 255))

        if self.warping:
            img = elastic_transform_image(img, image_dim*2, image_dim*0.1)

        img = np.expand_dims(img, -1)
        
        if self.rotation_angle > 0:
            random_angle = self.rotation_angle * (2 * np.random.random_sample() - 1)
            img = tv.transforms.functional.to_pil_image(img)
            img = tv.transforms.functional.rotate(img, random_angle, resample=PIL.Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        if not self.is_predict:
            target = np.zeros((image_dim, image_dim))
            if pid in self.boxes_by_pid_dict:
                for box in self.boxes_by_pid_dict[pid]:
                    x, y, w, h = box

                    x = int(round(x / self.rescale_factor))
                    y = int(round(y / self.rescale_factor))
                    w = int(round(w / self.rescale_factor))
                    h = int(round(h / self.rescale_factor))

                    # create mask over the boxes
                    target[y:y+h, x:x+w] = 255
                    target[target > 255] = 255

            target = np.expand_dims(target, -1)
            target = target.astype('uint8')

            if self.rotation_angle > 0:
                target = tv.transforms.functional.to_pil_image(target)
                target = tv.transforms.functional.rotate(target, random_angle, resample=PIL.Image.BILINEAR)

            if self.transform is not None:
                target = self.transform(target)

            return img, target, pid
        else:
            return img, pid

    def __len__(self):
        return len(self.pids)
