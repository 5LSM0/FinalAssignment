import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import wandb


def train_model(model, train_loader, val_loader, num_epochs=5, lr=0.01, patience=3):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize best_val_loss here
    best_val_loss = float('inf')

    # Create and open a text file
    with open('training_log.txt', 'w') as file:
        for epoch in range(num_epochs):
            # Initialize accuracy variables for each epoch
            total_correct_train, total_samples_train = 0, 0
            total_correct_val, total_samples_val = 0, 0

            # TRAINING
            model.train()
            running_train_loss = 0.0
            for inputs, masks in train_loader:
                # Move inputs and masks to the GPU
                inputs, masks = inputs.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                masks = (masks * 255)
                loss = criterion(outputs, masks.long().squeeze())  # .squeeze() - unable to convert a tensor to a 1D tensor
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                # Compute training accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct_train += (predicted == masks.long().squeeze()).sum().item()
                total_samples_train += masks.numel()

            train_epoch_loss = running_train_loss / len(train_loader)

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

                    # Compute validation accuracy
                    _, val_predicted = torch.max(val_outputs, 1)
                    total_correct_val += (val_predicted == val_masks.long().squeeze()).sum().item()
                    total_samples_val += val_masks.numel()

                epoch_val_loss = running_val_loss / len(val_loader)

            # Early stopping functionality
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_consecutive_epoch_without_improve = 0
            else:
                num_consecutive_epoch_without_improve += 1

            if num_consecutive_epoch_without_improve >= patience:
                # Save the model to the checkpoint file
                save_checkpoint(model, optimizer, epoch, best_val_loss)
                print(f"EARLY STOPPING INVOKED: Training halted after {num_consecutive_epoch_without_improve} epochs without improvement.")
                break

            # Calculate and print accuracy
            accuracy_train = total_correct_train / total_samples_train
            accuracy_val = total_correct_val / total_samples_val

            # Write to the log file
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {accuracy_val:.4f}\n')

def train_model_wandb(model, train_loader, val_loader, num_epochs=5, lr=0.01, patience=3):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Create and open a text file
    with open('training_log.txt', 'w') as file:
        for epoch in range(num_epochs):
            # Initialize accuracy variables for each epoch
            total_correct_train, total_samples_train = 0, 0
            total_correct_val, total_samples_val = 0, 0

            # TRAINING
            model.train()
            running_train_loss = 0.0
            for inputs, masks in train_loader:
                # Move inputs and masks to the GPU
                inputs, masks = inputs.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                masks = (masks * 255)
                loss = criterion(outputs, masks.long().squeeze())  # .squeeze() - unable to convert a tensor to a 1D tensor
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                # Compute training accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct_train += (predicted == masks.long().squeeze()).sum().item()
                total_samples_train += masks.numel()

            train_epoch_loss = running_train_loss / len(train_loader)

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

                    # Compute validation accuracy
                    _, val_predicted = torch.max(val_outputs, 1)
                    total_correct_val += (val_predicted == val_masks.long().squeeze()).sum().item()
                    total_samples_val += val_masks.numel()

                epoch_val_loss = running_val_loss / len(val_loader)

            # Early stopping functionality
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                num_consecutive_epoch_without_improve = 0
            else:
                num_consecutive_epoch_without_improve += 1

            if num_consecutive_epoch_without_improve >= patience:
                # Save the model to the checkpoint file
                save_checkpoint(model, optimizer, epoch, best_val_loss)
                print(f"EARLY STOPPING INVOKED: Training halted after {num_consecutive_epoch_without_improve} epochs without improvement.")
                break

            # Calculate and print accuracy
            accuracy_train = total_correct_train / total_samples_train
            accuracy_val = total_correct_val / total_samples_val

            # Log metrics to wandb
            wandb.log({
                'Epoch': epoch + 1,
                'Train Loss': train_epoch_loss,
                'Train Accuracy': accuracy_train,
                'Val Loss': epoch_val_loss,
                'Val Accuracy': accuracy_val
            })

            # Write to the log file
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {accuracy_val:.4f}\n')


def save_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint_path = 'model_checkpoint.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)