from torch.utils.data import Dataset
from utils import crop_patch,bicubic_downsample
import torchvision.transforms as transforms
import random

class SuperResolutionDataset(Dataset):
    
    def __init__(self, hr_images, patch_size, scale_factor, augment = None,norm=None):
        '''
        Parameters
        ----------
        hr_images : list
            List of high-resolution images.
        patch_size : int
            Size of the patches to be extracted.
        scale_factor : int
            Scale factor for downsampling.
        augment : callable, optional
            Augmentation function to apply to the images. 
        norm : callable, optional
            Normalization function to apply to the images.
        '''
        self.hr_images = hr_images
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.norm = norm
    
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        '''
        Parameters
        ----------
        idx : int
            Index of the image to be retrieved.
        Returns
        -------
        tuple
            Tuple containing the low-resolution and high-resolution patches.
        '''
        hr_image = self.hr_images[idx]
        if self.augment:
            hr_image = self.augment(hr_image)
        lr_image = bicubic_downsample(hr_image, self.scale_factor)
        lr_patch, hr_patch = crop_patch(lr_image, hr_image, self.patch_size, self.scale_factor)
        if self.norm:
            lr_patch = self.norm(lr_patch)
            hr_patch = self.norm(hr_patch)
        
        return lr_patch, hr_patch

