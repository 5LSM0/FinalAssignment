import numpy as np
from torchvision import transforms
import torch

def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""

    img = transforms.functional.resize(img, size=(256, 256), interpolation=transforms.InterpolationMode.LANCZOS)
    trans = transforms.Compose([transforms.ToTensor()])
    img = trans(img)
    img = img.unsqueeze(0)

    return img

def postprocess(prediction, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy









