# MIT License
# 
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# Mainly modified from:
# - https://github.com/hasibzunair/3D-image-classification-tutorial/blob/master/3D_image_classification.ipynb
# ==============================================================================


import os
import nibabel as nib
from scipy import ndimage
import numpy as np
import torch.utils.data as data
import torch
import random


def read_nifti_file(filepath):
    """Read and load volume"""

    # Read file
    scan = nib.load(filepath)

    # Get raw data
    scan = scan.get_fdata() # H x W x D
    return scan


def normalize(volume):
    """Normalize the volume

    CT scans store raw voxel intensity in Hounsfield units (HU). They range from -1024 to above 2000 in this dataset.
    Above 400 are bones with different radiointensity, so this is used as a higher bound.
    A threshold between -1000 and 400 is commonly used to normalize CT scans.
    Read more: https://en.wikipedia.org/wiki/Hounsfield_scale
    """
    min = -1000
    max = 400

    volume = volume.astype("float32")
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min) # ranged [0.0, 1.0]

    return volume


def resize_volume(img, desired_depth=64, desired_height=128, desired_width=128):
    """Resize across z-axis"""

    # Get current depth
    current_depth = img.shape[-1]
    current_height = img.shape[0]
    current_width = img.shape[1]

    # Compute depth factor
    depth = current_depth / desired_depth
    height = current_height / desired_height
    width = current_width / desired_width

    depth_factor = 1. / depth
    height_factor = 1. / height
    width_factor = 1. / width

    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)

    # Resize across z-axis
    img = ndimage.zoom(img, (height_factor, width_factor, depth_factor), order=1)

    return img


def process_scan(path, shape):
    """Read and resize volume"""

    # Read scan
    volume = read_nifti_file(path)

    # Normalize
    volume = normalize(volume)

    # Resize height, width and depth
    height, width, depth = shape
    volume = resize_volume(volume, desired_height=height, desired_width=width, desired_depth=depth)

    return volume


def scipy_rotate(volume):
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, reshape=False)
    volume[volume < 0] = 0
    volume[volume > 1] = 1
    return volume


class COVID19Dataset(data.Dataset):
    def __init__(self, image_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations

        normal = sorted([os.path.join(os.path.join(image_root, "CT-0"), f) for f in os.listdir(os.path.join(image_root, "CT-0")) if f.endswith('.nii.gz')])
        abnormal = sorted([os.path.join(os.path.join(image_root, "CT-23"), f) for f in os.listdir(os.path.join(image_root, "CT-23")) if f.endswith('.nii.gz')])

        normal = np.array([process_scan(path, self.trainsize) for path in normal])
        abnormal = np.array([process_scan(path, self.trainsize) for path in abnormal])

        normal_labels = np.array([0 for _ in range(len(normal))])
        abnormal_labels = np.array([1 for _ in range(len(abnormal))])

        seed = np.random.randint(2147483647)

        np.random.seed(seed)
        self.x = np.concatenate((abnormal, normal), axis=0)
        np.random.shuffle(self.x)

        np.random.seed(seed)
        self.y = np.concatenate((abnormal_labels, normal_labels), axis=0)
        np.random.shuffle(self.y)

        self.size = len(self.x)

        if self.augmentations:
            print('Using Augmentation')
            self.transform = scipy_rotate   
        else:
            print('no augmentation')
            self.transform = None           

    def __getitem__(self, index):
        
        image = self.x[index]
        label = [self.y[index]]
        
        if self.transform is not None:
            image = self.transform(image)

        image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
        label = torch.Tensor(label)

        return image, label

    def __len__(self):
        return self.size


def get_loader(dataset, batchsize, shuffle=True, num_workers=2, pin_memory=True):

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_dataset(image_root, trainsize, augmentation=False): return COVID19Dataset(image_root, trainsize, augmentation)


class predict_dataset:
    def __init__(self, image_path, predict_size):
        self.predict_size = predict_size
        self.image = process_scan(image_path, self.predict_size)

    def load_data(self):
        image = self.image
        image = torch.Tensor(image).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
        return image
