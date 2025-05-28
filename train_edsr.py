import torch
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset
from model import EDSR
from training import train_model
from utils import compute_mean_and_std, load_hr_images,get_transform
import os
import argparse
from FourierSpaceLoss import CombinedLoss
#C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/data

def main(args):
    ''' 
    Main function to train the EDSR model.
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    Returns
    -------
    None
    '''
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join(args.dataset_path, 'train')
    valid_path = os.path.join(args.dataset_path, 'valid')
    mean, std = compute_mean_and_std(image_folder=args.dataset_path)
    augment, norm = get_transform(mean=mean,std=std,type='train')
    _,norm_valid = get_transform(mean=mean,std=std,type='valid')
    train_hr_images = load_hr_images(train_path)
    valid_hr_images = load_hr_images(valid_path)
    train_set = SuperResolutionDataset(hr_images=train_hr_images, patch_size=96 if args.frequency_loss else 48, scale_factor=args.scale, augment=augment,norm=norm)
    valid_set = SuperResolutionDataset(hr_images= valid_hr_images,  patch_size=96 if args.frequency_loss else 48, scale_factor=args.scale, augment=None, norm=norm_valid)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    model = EDSR(scale_factor=args.scale, num_filters=args.num_filters, num_res_blocks=args.num_blocks).to(device)
    if args.pre_train and args.scale > 2:
        prev_model_path = 'models/frequency_beta05_alpha1/edsr_x2_freq.pth' if args.frequency_loss else 'models/general_edsr/base_edsr_x2.pth' #adjust accordingly
        if os.path.exists(prev_model_path):
            print(f"Loading weights from {prev_model_path}")
            state_dict = torch.load(prev_model_path, map_location=device) 
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("upsample")}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            print(f"Pretrained model not found at {prev_model_path}")
            print("Training from scratch")
    if args.frequency_loss:
        print("Using CombinedLoss with Fourier loss")
        loss_fn = CombinedLoss(l1_weight=1.0, fourier_weight=0.05, alpha=1, beta=0.5, apply_hann=True) #alpha=0.5
        save_path = args.save_path or f"models/edsr_x{args.scale}_frequency.pth"
    else:
        print("Using standard L1 loss")
        loss_fn = torch.nn.L1Loss()
        save_path = args.save_path or f"models/edsr_x{args.scale}.pth"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200000,400000], gamma=0.5)
    train_model(model, train_loader, valid_loader, loss_fn, optimizer, args.epochs, device, save_path, args.patience)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EDSR Model")

    parser.add_argument('--dataset_path', type=str, required=True, help='Root path to dataset containing "train/" and "valid/" folders')
    parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (e.g., 2, 4, 8)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model, falls back to scale and name if not set')
    parser.add_argument('--num_filters', type=int, default=256, help='Number of filters')
    parser.add_argument('--num_blocks', type=int, default=32, help='Number of residual blocks')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--pre_train', action='store_true', help='Use pretrained model')
    parser.add_argument('--frequency_loss', action='store_true', help='Use Fourier loss')
    args = parser.parse_args()
    main(args)