from PIL import Image
import random
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import cv2
mean = [0.4488288587982963, 0.4371381120257274, 0.4040372117187323] #values for DIV2K
std = [0.2841556154456293, 0.27009665280451317, 0.2920475073076829]

def get_transform(mean,std,type):
    '''
    Get data augmentation and normalization transforms.
    Parameters
    ----------
    mean : list
        Mean values for normalization.
    std : list
        Standard deviation values for normalization.
    type : str
        Type of transformation ('train' or 'valid').
    '''
    if type == 'train':
        augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.Lambda(lambda x: x),  # No rotation (identity)
            transforms.Lambda(lambda x: x.rotate(90, expand=True)),
            transforms.Lambda(lambda x: x.rotate(180, expand=True)),
            transforms.Lambda(lambda x: x.rotate(270, expand=True))]),
        ])
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return augment,norm
    elif type == 'valid':
        norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
        return None, norm
    else:
        print('no valid transformation found')
    return None,None    

def bicubic_downsample(image, scale_factor):
    '''
    Downsample the image using bicubic interpolation.
    Parameters
    ----------
    image : np.ndarray or PIL.Image
        Input image to be downsampled.
    scale_factor : int
        Downsampling factor.
    Returns
    -------
    PIL.Image
        Downsampled image.
    '''
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    width, height = image.size
    lr_image = image.resize((width // scale_factor, height // scale_factor), Image.BICUBIC)
    return lr_image

def crop_patch(lr_image, hr_image, patch_size=48, scale_factor=2):
    '''
    Crop a patch from the low-resolution and high-resolution images.
    Parameters
    ----------
    lr_image : PIL.Image
        Low-resolution image.
    hr_image : PIL.Image
        High-resolution image.
    patch_size : int
        Size of the patch to be cropped.
    scale_factor : int 
        Scale factor for the images.
    Returns
    -------
    lr_patch, hr_patch : tuple
        Cropped low-resolution and high-resolution patches.
    '''
    lr_width, lr_height = lr_image.size
    hr_patch_size = patch_size * scale_factor 
    x = random.randint(0, lr_width - patch_size)
    y = random.randint(0, lr_height - patch_size)
    lr_patch = lr_image.crop((x, y, x + patch_size, y + patch_size))
    x_hr = x * scale_factor
    y_hr = y * scale_factor
    hr_patch = hr_image.crop((x_hr, y_hr, x_hr + hr_patch_size, y_hr + hr_patch_size))
    return lr_patch, hr_patch

def load_hr_image(folder_path): #not used?
    files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            files.append(img_path)
    random_img = random.choice(files)
    image = cv2.imread(random_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
    return image

def denormalize(tensor, mean, std):
    '''
    Denormalize the tensor to the original image range.
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to be denormalized.
    mean : list
        Mean values for each channel.
    std : list
        Standard deviation values for each channel.
    Returns
    -------
    tensor : torch.Tensor
        Denormalized tensor.
    '''
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp(0, 1)

#def visualize_sample(model, valid_loader, device,mean,std): #not used?
#    '''
#    Visualize a sample from the validation set.
#    Parameters
#    ----------
#    model : torch.nn.Module
#        The model to be used for super-resolution.
#    valid_loader : DataLoader
#        DataLoader for the validation dataset.
#    device : torch.device
#        Device to run the model on.
#    mean : list
#        Mean values for normalization.
#    std : list
#        Standard deviation values for normalization.
#    Returns
#    -------
#    None
#    '''
#    print("Visualizing Super Resolution Results")
#    model.eval()  
#    lr_image, hr_image = random.choice(valid_loader.dataset)
#    lr_image = lr_image.unsqueeze(0).to(device)
#    with torch.no_grad():
#        sr_image = model(lr_image)
#    lr_image = lr_image.squeeze(0).cpu()
#    sr_image = sr_image.squeeze(0).cpu()
#    hr_image = hr_image.cpu()
#    lr_image = denormalize(lr_image, mean, std)
#    sr_image = denormalize(sr_image, mean, std)
#    hr_image = denormalize(hr_image, mean, std)
#    lr_image_np = lr_image.permute(1, 2, 0).numpy()  
#    sr_image_np = sr_image.permute(1, 2, 0).numpy()
#    hr_image_np = hr_image.permute(1, 2, 0).numpy()
#    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#    
#    axs[0].imshow(lr_image_np)
#    axs[0].set_title("Low-Resolution Input")
#    axs[0].axis('off')
#    axs[1].imshow(sr_image_np)
#    axs[1].set_title("Super-Resolved Output")
#    axs[1].axis('off')
#    axs[2].imshow(hr_image_np)
#    axs[2].set_title("High-Resolution Ground Truth")
#    axs[2].axis('off')
#    plt.show()


def compute_mean_and_std(image_folder):
    '''
    Compute the mean and standard deviation of the dataset.
    Parameters
    ----------
    image_folder : str
        Path to the folder containing images.
    Returns
    -------
    mean : list
        Mean values for each channel.
    std : list
        Standard deviation values for each channel.
    '''
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    sum_sq_r, sum_sq_g, sum_sq_b = 0.0, 0.0, 0.0
    num_pixels = 0
    transform = transforms.ToTensor()
    print("Computing mean and std for dataset")
    for root, _, files in os.walk(image_folder):
        for file in tqdm(files):
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert('RGB')
                
                img_tensor = transform(image)
                num_pixels += img_tensor.size(1) * img_tensor.size(2)
                sum_r += img_tensor[0].sum().item()
                sum_g += img_tensor[1].sum().item()
                sum_b += img_tensor[2].sum().item()
                sum_sq_r += (img_tensor[0] ** 2).sum().item()
                sum_sq_g += (img_tensor[1] ** 2).sum().item()
                sum_sq_b += (img_tensor[2] ** 2).sum().item()

    mean_r = sum_r / num_pixels
    mean_g = sum_g / num_pixels
    mean_b = sum_b / num_pixels
    std_r = (sum_sq_r / num_pixels - mean_r ** 2) ** 0.5
    std_g = (sum_sq_g / num_pixels - mean_g ** 2) ** 0.5
    std_b = (sum_sq_b / num_pixels - mean_b ** 2) ** 0.5
    return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]

def extract_patches(image,patch_size=256,overlap=64):
    '''
    Extract patches from the image with specified size and overlap.
    Parameters
    ----------
    image : np.ndarray
        Input image from which patches are to be extracted.
    patch_size : int
        Size of the patches to be extracted.
    overlap : int
        Overlap between patches.
    Returns
    -------
    patches : list
        List of extracted patches.
    patch_positions : list
        List of positions for each extracted patch.
    '''
    height,width,_ = image.shape
    stride = patch_size - overlap
    patches = []
    patch_positions = []
    for y in range(0,height-patch_size+1,stride):
        for x in range(0,width-patch_size+1,stride):
            patch = image[y:y+patch_size,x:x+patch_size,:]
            patches.append(patch)
            patch_positions.append((x,y))
    return patches,patch_positions

def vis_patches(lr_img, sr_img, gt_img,scale,save_path,max_vis=5):
    '''
    Visualize patches from low-resolution, super-resolved, and ground truth images.
    Parameters
    ----------
    lr_img : np.ndarray
        Low-resolution image.
    sr_img : np.ndarray
        Super-resolved image.
    gt_img : np.ndarray
        Ground truth image.
    scale : int
        Scale factor for super-resolution.
    save_path : str
        Path to save the visualization.
    max_vis : int
        Maximum number of patches to visualize.
    
    Returns
    -------
    None
    '''
    patch_size = 96
    lr_patch_size = patch_size // scale
    overlap = 0
    
    lr_patches, _ = extract_patches_unfold(lr_img, patch_size=lr_patch_size, overlap=overlap,hr_shape=None, scale=None)
    sr_patches, _ = extract_patches_unfold(sr_img, patch_size=patch_size , overlap=overlap,hr_shape=None, scale=None)
    gt_patches, _ = extract_patches_unfold(gt_img, patch_size=patch_size , overlap=overlap ,hr_shape=None, scale=None)
    
    num_vis = min(max_vis, lr_patches.shape[-1])
    num_cols = 4
    fig, axs = plt.subplots(num_vis, num_cols, figsize=(3.2 * num_cols, 3.0 * num_vis),constrained_layout=False)
    axs = np.atleast_2d(axs)
    
    num_patches_lr = lr_patches.shape[-1]
    num_patches_sr = sr_patches.shape[-1]
    num_patches_gt = gt_patches.shape[-1]
    total_patches = min(num_patches_lr, num_patches_sr, num_patches_gt)
    mid_start = total_patches // 3
    mid_end   = 2 * total_patches // 3
    center_indices = np.linspace(mid_start, mid_end - 1, num_vis, dtype=int)
    
    for row_idx, patch_idx in enumerate(center_indices):
        if row_idx == 0:
            axs[row_idx, 0].set_title("Low-Res",fontsize=25,pad=2)
            axs[row_idx, 1].set_title("Super-Resolved",fontsize=25,pad=2)
            axs[row_idx, 2].set_title("Ground Truth",fontsize=25,pad=2)
            axs[row_idx, 3].set_title("Residual",fontsize=25,pad=2)
        lr_patch = lr_patches[..., patch_idx].squeeze(0).float() / 255.0
        lr_patch_np = lr_patch.permute(1, 2, 0).cpu().numpy()
        sr_patch = sr_patches[..., patch_idx].squeeze(0).float() / 255.0
        sr_patch = sr_patch.permute(1, 2, 0).cpu().numpy()
        gt_patch = gt_patches[..., patch_idx].squeeze(0).float() / 255.0
        gt_patch = gt_patch.permute(1, 2, 0).cpu().numpy()
        residual = np.abs(gt_patch - sr_patch)
        residual *= 5.0 #due to residuals being small, we scale them up for better visibility
        residual = np.clip(residual, 0, 1)  # keep in displayable range
        
        axs[row_idx, 0].imshow(np.clip(lr_patch_np, 0, 1),interpolation='nearest')
        axs[row_idx, 0].axis("off")
        axs[row_idx, 1].imshow(sr_patch)
        axs[row_idx, 1].axis("off")
        if gt_patches is not None:
            gt_patch = gt_patches[..., patch_idx].squeeze(0).float() / 255.0
            gt_patch = gt_patch.permute(1, 2, 0).cpu().numpy()
            gt_patch = np.clip(gt_patch, 0, 1)
            axs[row_idx, 2].imshow(gt_patch)
            axs[row_idx, 2].axis("off")
            axs[row_idx, 3].imshow(residual)
            axs[row_idx, 3].axis("off")
    fig.subplots_adjust(
    left=0.01, right=0.99, top=0.99, bottom=0.01,
    wspace=0.01, hspace=0.02)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def super_resolve_patches(patches, model, device="cuda", vis=False, max_vis=5,save_path=None, gt_patches=None):
    '''
    Super-resolve patches using the provided model.
    Parameters
    ----------
    patches : torch.Tensor
        Input patches to be super-resolved.
    model : torch.nn.Module
        The model to be used for super-resolution.
    device : str
        Device to run the model on.
    vis : bool
        Whether to visualize the results.
    max_vis : int
        Maximum number of patches to visualize.
    save_path : str
        Path to save the visualization.
    gt_patches : torch.Tensor
        Ground truth patches for comparison.
    Returns
    -------
    sr_patches_torch : torch.Tensor
        Super-resolved patches as a tensor.
    '''
    model.to(device)
    model.eval()
    B, C, H, W, N = patches.shape 
    assert B == 1, "Batch size must be 1"
    sr_patches_list = []
    transform = transforms.Normalize(mean=mean, std=std)
    with torch.no_grad():
        for i in range(N):
            patch = patches[..., i]  
            patch = patch.squeeze(0)  
            patch = patch.float() / 255.0 
            patch_tensor = transform(patch).unsqueeze(0).to(device) 
            sr_patch_tensor = model(patch_tensor)  
            sr_patch_tensor = sr_patch_tensor.squeeze(0).cpu()
            sr_patch_tensor = denormalize(sr_patch_tensor,mean=mean,std=std)
            sr_patch = sr_patch_tensor.numpy().transpose(1, 2, 0)
            sr_patch = np.clip(sr_patch * 255.0, 0, 255).astype(np.uint8)
            sr_patches_list.append(sr_patch)

    if vis:
        num_vis = min(max_vis, len(sr_patches_list))
        num_cols = 3 if gt_patches is not None else 2
        fig, axs = plt.subplots(num_vis, num_cols, figsize=(3*num_cols, 2 * num_vis))
        axs = np.atleast_2d(axs)
        total_patches = patches.shape[-1]
        mid_start = total_patches // 3
        mid_end = 2 * total_patches // 3
        center_indices = np.linspace(mid_start, mid_end - 1, num_vis, dtype=int)

        for row_idx, patch_idx in enumerate(center_indices-1):
            lr_patch = patches[..., patch_idx].squeeze(0).float() / 255.0
            lr_patch_np = lr_patch.permute(1, 2, 0).cpu().numpy()
            sr_patch = sr_patches_list[patch_idx]
            axs[row_idx, 0].imshow(np.clip(lr_patch_np, 0, 1))
            axs[row_idx, 0].set_title(f"LR Patch {patch_idx}")
            axs[row_idx, 0].axis("off")
            axs[row_idx, 1].imshow(sr_patch)
            axs[row_idx, 1].set_title(f"SR Patch {patch_idx}")
            axs[row_idx, 1].axis("off")
            if gt_patches is not None:
                gt_patch = gt_patches[..., patch_idx].squeeze(0).float() / 255.0
                gt_patch = gt_patch.permute(1, 2, 0).cpu().numpy()
                gt_patch = np.clip(gt_patch, 0, 1)
                axs[row_idx, 2].imshow(gt_patch)
                axs[row_idx, 2].set_title(f"GT Patch {patch_idx}")
                axs[row_idx, 2].axis("off")
        plt.tight_layout(pad=1.0, w_pad=0.1, h_pad=0.5)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    sr_patches_np = np.stack(sr_patches_list, axis=0)
    sr_patches_torch = torch.from_numpy(sr_patches_np).float()
    sr_patches_torch = sr_patches_torch.permute(0, 3, 1, 2)
    sr_patches_torch = sr_patches_torch.unsqueeze(0).permute(0, 2, 3, 4, 1)
    return sr_patches_torch

def extract_patches_unfold(image,patch_size=128,hr_shape=None,overlap=32,scale=None):
    '''
    Extract patches from the image using unfold.
    Parameters
    ----------
    image : np.ndarray
        Input image from which patches are to be extracted.
    patch_size : int
        Size of the patches to be extracted.
    hr_shape : tuple
        Shape of the high-resolution image (height, width).
    overlap : int
        Overlap between patches.
    scale : int
        Scale factor for the images.
    Returns
    -------
    patches : torch.Tensor
        Extracted patches as a tensor.
    shape : tuple
        Shape of the patches and additional information.
    '''
    stride = patch_size - overlap
    H, W, C = image.shape
    if hr_shape is not None:
        H_hr, W_hr = hr_shape
        needed_lr_h = math.ceil(H_hr / scale)
        needed_lr_w = math.ceil(W_hr / scale)
        if H < needed_lr_h:
            extra_h = needed_lr_h - H
            image = np.pad(image, ((0, extra_h), (0, 0), (0, 0)), mode='constant')
            H += extra_h
        if W < needed_lr_w:
            extra_w = needed_lr_w - W
            image = np.pad(image, ((0, 0), (0, extra_w), (0, 0)), mode='constant')
            W += extra_w
            
    pad_bottom, pad_right = 0, 0
    
    if H < patch_size:
        pad_bottom = patch_size - H
    else:
        remainder_h = (H - patch_size) % stride
        if remainder_h != 0:
            pad_bottom = stride - remainder_h
    if W < patch_size:
        pad_right = patch_size - W
    else:
        remainder_w = (W - patch_size) % stride
        if remainder_w != 0:
            pad_right = stride - remainder_w
            
    image_padded = np.pad(
        image,
        pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
        mode='constant'
    )
    image_tensor = torch.from_numpy(image_padded).permute(2, 0, 1).unsqueeze(0).float()
    patches = F.unfold(image_tensor, kernel_size=patch_size, stride=stride)
    C = image_tensor.shape[1]
    num_patches = patches.shape[-1]
    patches = patches.view(1, C, patch_size, patch_size, num_patches)
    shape = (C, patch_size, patch_size, num_patches, pad_bottom, pad_right, H, W,hr_shape,scale)
    return patches,shape

def merge_patches_fold(sr_patches,lr_image_shape,patch_size,overlap):
    ''' 
    Merge the super-resolved patches into a single image using fold.
    Parameters
    ----------
    sr_patches : torch.Tensor
        Super-resolved patches to be merged.
    lr_image_shape : tuple
        Shape of the low-resolution image (height, width).
    patch_size : int
        Size of the patches.
    overlap : int
        Overlap between patches.
    Returns
    -------
    sr_stitched : torch.Tensor
        Merged super-resolved image.
    '''
    B, C, up_patch_size, up_patch_size2, N = sr_patches.shape
    assert B == 1 and up_patch_size == up_patch_size2
    _, _, _, _, pad_bottom, pad_right, H_orig, W_orig,hr_shape,scale = lr_image_shape
    lr_h_padded = H_orig + pad_bottom
    lr_w_padded = W_orig + pad_right
    hr_h_padded = lr_h_padded * scale
    hr_w_padded = lr_w_padded * scale
    sr_stride = (patch_size - overlap) * scale

    sr_patches_2d = sr_patches.view(B,C * up_patch_size * up_patch_size,N)
    sr_summed = F.fold(sr_patches_2d, output_size=(hr_h_padded, hr_w_padded),kernel_size=up_patch_size,stride=sr_stride)

    ones = torch.ones_like(sr_patches_2d)
    weight_map = F.fold(ones,output_size=(hr_h_padded, hr_w_padded),kernel_size=up_patch_size,stride=sr_stride)
    sr_stitched_padded = sr_summed / (weight_map + 1e-8)
    hr_h_original = H_orig * scale
    hr_w_original = W_orig * scale
    sr_stitched = sr_stitched_padded[..., :hr_h_original, :hr_w_original]
    if hr_shape is not None:
        H_hr, W_hr = hr_shape
        if sr_stitched.shape[-2] > H_hr:
            sr_stitched = sr_stitched[..., :H_hr, :]
        if sr_stitched.shape[-1] > W_hr:
            sr_stitched = sr_stitched[..., :,:W_hr]
    return sr_stitched

#def load_single_image(image_path): #not used?
#    image = cv2.imread(image_path)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    return image

def visualize_patches_grid(patches, unfolded_shape):
    '''
    Visualize patches in a grid format.
    Parameters
    ----------
    patches : torch.Tensor
        Patches to be visualized.
    unfolded_shape : tuple
        Shape of the unfolded patches.
    Returns
    -------
    None
    '''
    C, H_patch, W_patch, num_patches_total = unfolded_shape
    patches_np = patches.view(C, H_patch, W_patch, num_patches_total).permute(3, 1, 2, 0).numpy()
    patches_np = np.clip(patches_np, 0, 255).astype(np.uint8)
    grid_cols = math.ceil(math.sqrt(num_patches_total))  
    grid_rows = math.ceil(num_patches_total / grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    axes = axes.flatten()  
    for i in range(num_patches_total):
        axes[i].imshow(patches_np[i])
        axes[i].set_title(f"Patch {i}")
        axes[i].axis("off")
    for i in range(num_patches_total, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
    
def load_hr_images(folder_path):
    hr_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert('RGB')  
            hr_images.append(image)
    print(f'loaded images from {folder_path}')
    return hr_images