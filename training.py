import torch
import time
from tqdm import tqdm
import logging
import os

def train(model, train_loader, criterion, optimizer, device):
    '''
    Train the model for one epoch.
    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for the model.
    device : torch.device
        Device to run the model on (CPU or GPU).
    Returns
    -------
    epoch_loss : float
        Average loss for the epoch.
    '''
    model.train()
    running_loss = 0.0  
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for lr_patches, hr_patches in train_loader:
        lr_patches = lr_patches.to(device)
        hr_patches = hr_patches.to(device)
        optimizer.zero_grad()  
        outputs = model(lr_patches)
        loss = criterion(outputs, hr_patches)
        loss.backward()
        optimizer.step()
        #scheduler.step()#
        running_loss += loss.item() * lr_patches.size(0)
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, valid_loader, criterion, device):
    '''
    Validate the model on the validation dataset.
    Parameters
    ----------
    model : torch.nn.Module
        The model to be validated.
    valid_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.Module
        Loss function.
    device : torch.device
        Device to run the model on (CPU or GPU).
    Returns
    -------
    epoch_loss : float
        Average loss for the epoch.
    '''
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(valid_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for lr_patches, hr_patches in valid_loader:
            lr_patches = lr_patches.to(device)
            hr_patches = hr_patches.to(device)
            outputs = model(lr_patches)
            loss = criterion(outputs, hr_patches)
            running_loss += loss.item() * lr_patches.size(0)
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
    epoch_loss = running_loss / len(valid_loader.dataset)
    return epoch_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler("training.log", mode='w')])

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, save_path=None, patience=10):
    '''
    Train the model with early stopping based on validation loss.
    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    valid_loader : DataLoader
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer for the model.
    num_epochs : int
        Number of epochs to train the model.
    device : torch.device
        Device to run the model on (CPU or GPU).
    save_path : str, optional
        Path to save the model.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped.
    '''
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training
        logging.info(f"Epoch {epoch}/{num_epochs}: Starting training...")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # Validation
        logging.info(f"Epoch {epoch}/{num_epochs}: Starting validation...")
        val_loss = validate(model, valid_loader, criterion, device)
        
        end_time = time.time()
        epoch_time = end_time - start_time

        logging.info(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            if save_path:
                logging.info(f"Validation loss improved to {val_loss:.4f}. Saving model to {save_path}")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            logging.info(f"No improvement in validation loss for {counter}/{patience} epochs.")
        if counter >= patience:
            logging.info(f"Stopping training after {counter} epochs without improvement.")
            break
    logging.info("Training completed.")