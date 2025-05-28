import torch
import torch.nn as nn
import torch.fft

class FourierLoss(nn.Module):
    '''Fourier Loss for comparing two images in the Fourier domain.'''
    
    def __init__(self,alpha=1.0,beta=1.0,apply_hann=False):
        super(FourierLoss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.apply_hann = apply_hann
        self.hann_window = None
    
    def forward(self, pred, target):
        '''Compute the Fourier loss between the predicted and target images.
        Parameters
        ----------
        pred : torch.Tensor
            Predicted image tensor.
        target : torch.Tensor
            Target image tensor.
        Returns
        -------
        total_loss: torch.Tensor
            Computed Fourier loss.
        '''
        pred = pred.float()
        target = target.float()
        
        if self.apply_hann:
            hann_window = self.get_hann_window(pred.shape[-2:]).to(pred.device)
            pred *= hann_window
            target *= hann_window
            
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        U,V = pred_fft.shape[-2:]
        pred_fft = pred_fft[:, :, :U//2, :]
        target_fft = target_fft[:, :, :U//2, :]
        
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        amp_loss = (2 / (U * V)) * torch.sum(torch.abs(pred_amp - target_amp))
        phase_loss = (2 / (U * V)) * torch.sum(torch.abs(pred_phase - target_phase))
        total_loss = self.alpha * amp_loss + self.beta * phase_loss
        return total_loss

    def get_hann_window(self,size):
        '''Generate a 2D Hann window.
        Parameters
        ----------
        size : tuple
            Size of the window (height, width).
        Returns
        -------
        hann2d : torch.Tensor
            2D Hann window tensor.
        '''
        h,w = size
        hann_1d_h = torch.hann_window(h,periodic=False)
        hann_1d_w = torch.hann_window(w,periodic=False)
        hann2d = torch.outer(hann_1d_h,hann_1d_w)
        return hann2d.unsqueeze(0).unsqueeze(0)
    
class CombinedLoss(nn.Module):
    '''Combined loss function that includes L1 loss and Fourier loss.'''
    def __init__(self,l1_weight=1.0,fourier_weight=1.0,alpha=1.0,beta=1.0,apply_hann=False):
        super(CombinedLoss,self).__init__()
        self.l1_weight = l1_weight
        self.fourier_weight = fourier_weight
        self.l1_loss = nn.L1Loss()
        self.fourier_loss = FourierLoss(alpha,beta,apply_hann)
        
    def forward(self,pred,target):
        l1_loss = self.l1_loss(pred,target)
        fourier_loss = self.fourier_loss(pred,target)
        total_loss = self.l1_weight * l1_loss + self.fourier_weight * fourier_loss
        return total_loss