"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
from model import Model
from model_executables import train_model_wandb
import augmentations as A
from torchvision.datasets import Cityscapes
from torch.utils.data import ConcatDataset, DataLoader, random_split
from argparse import ArgumentParser
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


import wandb

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimizer")
    parser.add_argument("--wandb_name", type=str, default="Default-UNet-with-Validation", help="Name of the wandb log")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # Define the transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
        transforms.Resize((256,256), antialias=True),  # Resize the input image to the given size
    ])

    # Define a list of transformations
    augment_tranmforms = [A.Resize((256, 256)), # This resize is to get a reference when cropping
                        A.RandomHorizontalFlip(),
                        A.RandomCropWithProbability(200, 0.6),
                        A.RandomRotation(degrees=(-35, 35)),
                        A.Resize((256, 256)), # this resize is to make sure that all the output images have intened size
                        A.ToTensor()]

    # Instanciate the Compose class with the list of transformations
    aug_transforms = A.Compose(augment_tranmforms)

    # Create transformed train dataset
    training_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=data_transforms, target_transform=data_transforms)

    # Create the augmented training dataset
    augmented_training_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transforms=aug_transforms)

    # Combine the datasets
    combined_dataset = ConcatDataset([training_dataset, augmented_training_dataset])

    # Determine the lengths of the training and validation sets
    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size  # 20% for validation

    # Shuffle and Split the combined dataset 
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10,
                                    pin_memory=True if torch.cuda.is_available() else False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=10,
                                    pin_memory=True if torch.cuda.is_available() else False)

    # Instanciate the model
    UNet_model = Model()

    # Move the model to the GPu if avaliable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    UNet_model = UNet_model.to(device)

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(UNet_model.parameters(), lr=args.lr)


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="5LSM0-WB-UNet-train",
        name="Default-UNet-with-Validation",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": "UNet",
        "dataset": "Cityspace",
        "epochs": args.epochs,
        }
    )

    # Train the instanciated model
    train_model_wandb(model=UNet_model, train_loader=train_dataloader,
                    val_loader=val_dataloader, num_epochs=args.epochs,
                    criterion=criterion, optimizer=optimizer, patience=4)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
    
    # visualize some results

    pass

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
