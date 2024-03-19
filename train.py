"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
from model import Model
from model_executables import train_model_wandb
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


import wandb

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # Define the transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
        transforms.Resize((256,256))
    ])

    # Create transformed train dataset
    training_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=data_transforms, target_transform=data_transforms)

    # Determine the lengths of the training and validation sets
    total_size = len(training_dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    val_size = total_size - train_size  # 20% for validation

    # Shuffle and Split the train dataset
    training_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [train_size, val_size])

    # Create Training and Validation DataLoaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=8,
                                            pin_memory=True if torch.cuda.is_available() else False)

    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=8,
                                            pin_memory=True if torch.cuda.is_available() else False)

    # Instanciate the model
    UNet_model = Model()

    # Move the model to the GPu if avaliable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    UNet_model = UNet_model.to(device)

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(UNet_model.parameters(), lr=0.01)


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="5LSM0-WB-UNet-train",
        name="Default-UNet-with-Validation",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "UNet",
        "dataset": "Cityspace",
        "epochs": 70,
        }
    )

    # Train the instanciated model
    train_model_wandb(model=UNet_model, train_loader=train_loader, val_loader=val_loader, num_epochs=100, criterion=criterion, optimizer=optimizer, patience=4)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
    
    # visualize some results

    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
