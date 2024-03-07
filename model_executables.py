import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os



def train_model(model, train_loader, val_loader, num_epochs=5, lr=0.01, patience=3):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize best_val_loss here
    best_val_loss = float('inf')  

    for epoch in range(num_epochs):
        # TRAINING 
        model.train()
        running_train_loss = 0.0
        for inputs, masks in train_loader:
            # Move inputs and masks to the GPU
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            masks = (masks * 255)
            loss = criterion(outputs, masks.long().squeeze()) # .squeeze() - unable to convert a tensor to a 1D tensor
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_loss = running_train_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # VALIDATION
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_masks in val_loader:
                # Move inputs and masks to the GPU
                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)

                val_outputs = model(val_inputs)
                val_masks = (val_masks * 255)
                val_loss = criterion(val_outputs, val_masks.long().squeeze())

                running_val_loss += val_loss.item()

            epoch_val_loss = running_val_loss / len(val_loader)
            print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_val_loss:.4f}')

            # Early stopping functionality
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_consecutive_epoch_without_improve = 0 
            else:
                num_consecutive_epoch_without_improve += 1
            
            if num_consecutive_epoch_without_improve >= patience:
                # Save the model to the checkpoint file
                save_checkpoint(model, optimizer, epoch, best_val_loss)
                print(f"EARLY STOPPING INVOKED: Training haulted after {num_consecutive_epoch_without_improve} epochs without improvement.")
                break

        print("Training completed.")
    

def save_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint_path = 'model_checkpoint.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)