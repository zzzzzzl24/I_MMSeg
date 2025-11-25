import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, image1, image2, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, image1, image2, label


def random_rotate(image, image1, image2, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, image1, image2, label


def add_gaussian_noise(image, image1, image2, label, mean=0, std=0.2):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image1 = image1 + noise
    image2 = image2 + noise
    image = np.clip(image, 0, 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)  
    return image, image1, image2, label



class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, image1, image2, label = sample['image'], sample['image1'], sample['image2'], sample['label']

        if random.random() > 0.5:
            image, image1, image2, label = random_rot_flip(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = random_rotate(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = add_gaussian_noise(image, image1, image2, label)

        if random.random() > 0.5:
            shift_x = random.randint(-5, 5)  
            shift_y = random.randint(-5, 5)  
            image = np.roll(image, (shift_x, shift_y), axis=(0, 1))
            image1 = np.roll(image1, (shift_x, shift_y), axis=(0, 1))
            image2 = np.roll(image2, (shift_x, shift_y), axis=(0, 1))
            label = np.roll(label, (shift_x, shift_y), axis=(0, 1))

        # Random brightness or contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2) 
            image = np.clip(image * factor, 0, 255)
            image1 = np.clip(image1 * factor, 0, 255)
            image2 = np.clip(image2 * factor, 0, 255)

        # Random gamma correction
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / 255.0, gamma) * 255.0
            image1 = np.power(image1 / 255.0, gamma) * 255.0
            image2 = np.power(image2 / 255.0, gamma) * 255.0
        
        # Random crop (resize) or padding
        if random.random() > 0.5:
            top = random.randint(0, image.shape[0] // 5)
            left = random.randint(0, image.shape[1] // 5)
            image = image[top:top + self.output_size[0], left:left + self.output_size[1]]
            image1 = image1[top:top + self.output_size[0], left:left + self.output_size[1]]
            image2 = image2[top:top + self.output_size[0], left:left + self.output_size[1]]
            label = label[top:top + self.output_size[0], left:left + self.output_size[1]]
            
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  
            image1 = zoom(image1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image2 = zoom(image2, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image1 = torch.from_numpy(image1.astype(np.float32)).unsqueeze(0)
        image2 = torch.from_numpy(image2.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'image1':image1, "image2":image2, 'label': label.long()}
        return sample


class Myops_dataset(Dataset):
    def __init__(self, base_dir, base_dir1, base_dir2, list_dir, split, transform=None):
        """
            base_dir: bSSFP modal
            base_dir1: LGE modal
            base_dir2: T2w modal
        """
        self.transform = transform  
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.data_dir1 = base_dir1
        self.data_dir2 = base_dir2

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data_path_1 = os.path.join(self.data_dir1, slice_name+'.npz')
            data_path_2 = os.path.join(self.data_dir2, slice_name+'.npz')
            data = np.load(data_path)
            data_1 = np.load(data_path_1)
            data_2 = np.load(data_path_2)
            image, label = data['image'], data['label']
            image1 = data_1['image']
            image2 = data_2['image']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            filepath_1 = self.data_dir1 + "/{}.npy.h5".format(vol_name)
            filepath_2 = self.data_dir2 + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            data1 = h5py.File(filepath_1)
            data2 = h5py.File(filepath_2)
            image, label = data['image'][:], data['label'][:]
            image1 = data1['image'][:]
            image2 = data2['image'][:]

        sample = {'image': image, 'image1':image1, "image2":image2, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
