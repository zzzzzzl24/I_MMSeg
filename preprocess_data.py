import os
import time
import argparse
import h5py
import nibabel as nib
import numpy as np

from glob import glob
from tqdm import tqdm
from icecream import ic
from pathlib import Path

# The directory in which the current script resides, such as /home/fangdg/I_MMSeg
PROJECT_ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                   default= f'{PROJECT_ROOT}/MyoPS380_dataset/Raw_data', help='path for raw data')
parser.add_argument('--dst_path', type=str,
                   default= f'{PROJECT_ROOT}/MyoPS380_dataset/Process_data', help='root dir for data')
parser.add_argument('--use_normalize', action='store_true', default=True,
                   help='use normalize')
args = parser.parse_args()

def preprocess_train_image(image_files: str, label_files: str, dst_path:str, modality, index) -> None:
    os.makedirs(f"{dst_path}/{modality}/train_npz", exist_ok=True)

    a_min, a_max = 0, 255
    b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        name = image_file.split('/')[-1].replace(".nii.gz","")

        if str(name) in file_test:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape
        for dep in range(D):
            save_path = f"{dst_path}/{modality}/train_npz/{name}_slice{dep:03d}.npz"
            np.savez(save_path, label=label_data[:,:,dep], image=image_data[:,:,dep])
            if index == 0:
                with open(f"{PROJECT_ROOT}/list/train.txt", "a") as file:
                    file.write(f"{name}_slice{dep:03d}\n")
    pbar.close()


def preprocess_valid_image(image_files: str, label_files: str, dst_path:str, modality, index) -> None:
    os.makedirs(f"{dst_path}/{modality}/test_vol_h5", exist_ok=True)

    a_min, a_max = 0, 255

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        name = image_file.split('/')[-1].replace(".nii.gz","")

        if str(name) not in file_test:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        save_path = f"{dst_path}/{modality}/test_vol_h5/{name}.npy.h5"
        f = h5py.File(save_path, 'w')
        f['image'] = image_data
        f['label'] = label_data
        f.close()

        if index == 0:
            with open(f"{PROJECT_ROOT}/list/test_vol.txt", "a") as file:
                file.write(f"{name}\n")
    pbar.close()


if __name__ == "__main__":
    src_path = args.src_path
    dst_path = args.dst_path
    os.makedirs(f"{dst_path}", exist_ok=True)
    index=0
    for modality in ["bSSFP", "LGE", "T2w"]:
        data_root = src_path + f"/{modality}"
        filenames = []
        image_files = sorted(glob(f"{data_root}/*.nii.gz"))
        label_files = sorted(glob(f"{src_path}/label/*.nii.gz"))
        for filename in image_files:
            filename = filename.split('/')[-1].replace(".nii.gz","")
            if index == 0 :
                with open(f"{PROJECT_ROOT}/list/all.txt", "a") as file:
                    file.write(f"{filename}.npy.h5\n")
            filenames.append(filename)
        file_test = filenames[0:380:5]
        preprocess_train_image(image_files, label_files, dst_path, modality, index)
        preprocess_valid_image(image_files, label_files, dst_path, modality, index)
        index+=1
