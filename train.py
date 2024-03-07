"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
# from model import Model
from torchvision.datasets import Cityscapes
from argparse import ArgumentParser
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # Define set of transforms
    # Define the transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
        transforms.Resize((256,256),antialias=True)
    ])

    # data loading
    original_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=None, target_transform=None)
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=data_transforms, target_transform=data_transforms)

    # Extract info about the original dataset
    num_images_original = len(original_dataset)
    image_original, _ = original_dataset[0]
    image_size_original = image_original.width, image_original.height
    num_classes_original = original_dataset.classes

    # Extract info about the transformed dataset
    num_images = len(dataset)
    image_size = dataset[0][0].size()
    num_classes = dataset.classes

    # Create and open a text file
    with open('dataset_info.txt', 'w') as file:
        # Original/untransformed dataset info
        file.write("Dataset state after transform:\n")
        file.write(f"Number of images: {num_images_original}\n")
        file.write(f"Image size: {image_size_original}\n")
        file.write(f"Number of classes: {num_classes_original}\n")
        # Dataset info AFTER applying the transforms
        file.write("\n")
        file.write("Dataset state after transform:\n")
        file.write(f"Number of images: {num_images}\n")
        file.write(f"Image size: {image_size}\n")
        file.write(f"Number of classes: {num_classes}\n")

    # visualize example images
    
    plt.subplots(4, 2, figsize=(10, 15))
    
    for i in range(4):
        img, lbl = dataset[i]
    
        img_np = img.permute(1, 2, 0)
        lbl_np = lbl.permute(1, 2, 0)
    
        plt.subplot(4, 2, 2 * i + 1)
        plt.imshow(img_np)
        plt.title(f'RGB Image {i+1}')
    
        plt.subplot(4, 2, 2 * i + 2)
        plt.imshow(lbl_np)  # Adjust the colormap as needed
        plt.title(f'Semantic Segmentation Label {i+1}')
        
    # Save a figure to a PNG format file
    plt.savefig('Cityspace-test-vis.png')
    
    plt.show()

    # define model
    # model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)


    # training/validation loop


    # save model


    # visualize some results

    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
