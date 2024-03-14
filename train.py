"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch
from model import Model
from model_executables import train_model_wandb_noval
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
        transforms.Resize((256,256),antialias=True)
    ])

    # Create transformed train dataset
    training_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=data_transforms, target_transform=data_transforms)

    # # Create transformed train dataset
    # validation_dataset = Cityscapes(args.data_path, split='val', mode='fine', target_type='semantic', transform=data_transforms, target_transform=data_transforms)

    # Extract info about the transformed training dataset
    train_num_images = len(training_dataset)
    train_image_size = training_dataset[0][0].size()
    train_num_classes = training_dataset.classes

    # Extract info about the transformed validation dataset
    # val_num_images = len(validation_dataset)
    # val_image_size = validation_dataset[0][0].size()
    # val_num_classes = validation_dataset.classes

    # Create and open a text file
    with open('dataset_info.txt', 'w') as file:
        # Transformed training dataset info
        file.write("Training Dataset state after transform:\n")
        file.write(f"Number of images: {train_num_images}\n")
        file.write(f"Image size: {train_image_size}\n")
        file.write(f"Number of classes: {train_num_classes}\n")
        # Transformed validation dataset info
        # file.write("\n")
        # file.write("Validation Dataset state after transform:\n")
        # file.write(f"Number of images: {val_num_images}\n")
        # file.write(f"Image size: {val_image_size}\n")
        # file.write(f"Number of classes: {val_num_classes}\n")

    # visualize example images
    
    plt.subplots(4, 2, figsize=(10, 15))
    
    for i in range(4):
        img, lbl = training_dataset[i]
    
        img_np = img.permute(1, 2, 0)
        lbl_np = lbl.permute(1, 2, 0)
    
        plt.subplot(4, 2, 2 * i + 1)
        plt.imshow(img_np)
        plt.title(f'RGB Image {i+1}')
    
        plt.subplot(4, 2, 2 * i + 2)
        plt.imshow(lbl_np)  # Adjust the colormap as needed
        plt.title(f'Semantic Segmentation Label {i+1}')
        
    # Save a figure to a PNG format file
    # plt.savefig('Cityspace-test-vis.png')
    
    plt.show()
    
    # Create Training and Validation DataLoaders
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=8,
                                            pin_memory=True if torch.cuda.is_available() else False)

    # val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=True, num_workers=2,
    #                                         pin_memory=True if torch.cuda.is_available() else False)

    # Instanciate the model
    UNet_model = Model()

    # Move the model to the GPu if avaliable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    UNet_model = UNet_model.to(device)

    # define optimizer and loss function (don't forget to ignore class index 255)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="5LSM0-WB-UNet-train",
        name="Fixed-class-labeling-UNet-train",
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "UNet",
        "dataset": "Cityspace",
        "epochs": 50,
        }
    )

    # Train the instanciated model
    train_model_wandb_noval(model=UNet_model, train_loader=train_loader, num_epochs=50, lr=0.01, patience=4)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
    
    # visualize some results

    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
