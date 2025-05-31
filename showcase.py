from utils import bicubic_downsample,extract_patches_unfold,super_resolve_patches,merge_patches_fold,load_hr_image,vis_patches
#from lpips import calc_lpips
import lpips
from sharpness_measure import calc_FM
from model import EDSR
import torch
import argparse
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def create_comparison_plot_matplot(hr_img, lr_img, sr_img, save_path, lpips_fn, scale, device='cuda'):
    '''
    Create a comparison plot of the low-res, super-resolved, and high-res images, as well as calculation of evaluation metrics.
    Parameters
    ----------
    hr_img : np.ndarray or PIL.Image
        High-resolution image.
    lr_img : np.ndarray or PIL.Image
        Low-resolution image.
    sr_img : np.ndarray or PIL.Image
        Super-resolved image.
    save_path : str
        Path to save the comparison plot.
    lpips_fn : callable
        LPIPS loss function.
    scale : int
        Upscaling factor.
    device : str
        Device to use (cuda or cpu).
    '''
    if isinstance(hr_img, np.ndarray):
        hr_img = Image.fromarray(hr_img)
    if isinstance(lr_img, np.ndarray):
        lr_img = Image.fromarray(lr_img)
    if isinstance(sr_img, np.ndarray):
        sr_img = Image.fromarray(sr_img)
    
    hr_np = np.array(hr_img)
    lr_up = lr_img.resize(hr_img.size, resample=Image.NEAREST)
    lr_np = np.array(lr_up)
    sr_np = np.array(sr_img)

    psnr_val = psnr(hr_np, sr_np, data_range=255)
    ssim_val = ssim(hr_np, sr_np, data_range=255, win_size=7, channel_axis=-1)
    # Convert images to PyTorch tensors
    hr_t = torch.from_numpy(hr_np).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    sr_t = torch.from_numpy(sr_np).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
    # scaling of img into LPIPS range ([-1,1])
    hr_t = hr_t * 2.0 - 1.0
    sr_t = sr_t * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = lpips_fn(sr_t, hr_t).item()    
    sharpness_val = calc_FM(sr_np)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    axs[0].imshow(lr_np,interpolation='nearest')
    axs[0].set_title("Low-Res",fontsize=25)
    axs[0].axis("off")
    axs[1].imshow(sr_np)
    axs[1].set_title(f"Super-Resolved x{scale}\nPSNR: {psnr_val:.2f}  SSIM: {ssim_val:.4f} LPIPS: {lpips_val:.8f}  FM: {sharpness_val:.4f}",fontsize=25)
    axs[1].axis("off")
    axs[2].imshow(hr_np)
    axs[2].set_title("Ground Truth",fontsize=25)
    axs[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def main(args):
    '''
    Main function to load the EDSR model, process images, and create comparison plots.
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    '''
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = EDSR(scale_factor=args.scale, num_filters=args.num_filters, num_res_blocks=args.num_blocks).to(device)
    model_path = args.model_path if args.model_path else f"models/edsr_x{args.scale}.pth"
    state_dict = torch.load(model_path, map_location=device) 
    model.load_state_dict(state_dict)
    print(f"Model successfully loaded from {model_path}")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    
    gt_dir = os.path.join(args.img_path, 'ground_truth')
    hr_img_gt = load_hr_image(gt_dir)
    lr_image = bicubic_downsample(hr_img_gt, scale_factor=args.scale)
    lr_img_path = os.path.join(args.img_path,'low_res',args.lr_name)
    lr_image.save(lr_img_path)
    
    lr_image_np = np.array(lr_image)
    patches,unfold_shape = extract_patches_unfold(lr_image_np,patch_size=128,overlap=32,hr_shape=hr_img_gt.shape[:2],scale = args.scale)
    gt_patches, _ = extract_patches_unfold(hr_img_gt, patch_size=128*args.scale, overlap=32*args.scale, hr_shape=hr_img_gt.shape[:2], scale=1)
    sr_patches = super_resolve_patches(patches, model, device="cuda", vis=False,save_path=os.path.join(args.img_path,'upscaled',args.patch_vis),gt_patches=gt_patches)
    final_sr = merge_patches_fold(sr_patches,lr_image_shape=unfold_shape,patch_size=128,overlap=32)
    sr_image = final_sr.squeeze(0)  
    sr_image = sr_image.permute(1, 2, 0).cpu().numpy() 
    if sr_image.max() <= 1.0:
        sr_image = (sr_image * 255.0).clip(0, 255)
    sr_image = sr_image.astype(np.uint8)
    sr_image = Image.fromarray(sr_image)
    sr_image.save(os.path.join(args.img_path,'upscaled',args.out_name))
    vis_patches(lr_img=np.array(lr_image),sr_img=np.array(sr_image),gt_img=np.array(hr_img_gt),scale=args.scale,save_path=os.path.join(args.img_path, 'upscaled', args.patch_vis),max_vis=5)
    create_comparison_plot_matplot(hr_img_gt,lr_image,sr_image, os.path.join(args.img_path,'upscaled',args.comp_name), lpips_fn=loss_fn_vgg, scale=args.scale)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EDSR Model")
    parser.add_argument('--model_path', type=str, default=None, help='Path to model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--img_path', type=str, default='imgs', help='Path to images')
    parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (e.g., 2, 4, 8)')
    parser.add_argument('--num_filters', type=int, default=256, help='Number of filters')
    parser.add_argument('--num_blocks', type=int, default=32, help='Number of residual blocks')
    #parser.add_argument('--dataset_path', type=str, default='C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/data', help='Root path to dataset containing "train/" and "valid/" folders')
    parser.add_argument('--comp_name', type=str, default='comparison.png', help='Name of the comparison image')
    parser.add_argument('--out_name', type=str, default='output.png', help='Name of the output image')
    parser.add_argument('--lr_name', type=str, default='lr_image.png', help='Name of the low-res image')
    parser.add_argument('--patch_vis',type=str, default='patches.png', help='Name of the patch visualization image')
    args = parser.parse_args()
    main(args)

