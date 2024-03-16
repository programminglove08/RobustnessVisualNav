import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.io import read_image
from io import BytesIO
import base64
import io
def load_image(image_path, preprocess, device):
    """
    Loads and preprocesses an image from a given path.
    Args:
        image_path (str): Path to the image file.
        preprocess (function): Preprocessing function (e.g., from CLIP).
        device (torch.device): Device to load the image onto.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)

def preprocess_image(image_data, preprocess, device):
    """
    Loads and preprocesses an image from a given image data.
    Args:
        image_data (bytes): Image Data.
        preprocess (function): Preprocessing function (e.g., from CLIP).
        device (torch.device): Device to load the image onto.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    #image = Image.open(io.BytesIO(image_data))
    return preprocess(image_data).unsqueeze(0).to(device)


def save_image_file(image_tensor, output_dir, output_filename):
    """
    Saves a single image tensor to a file in the specified directory with the given filename.
    
    Args:
        image_tensor (torch.Tensor): The image tensor to save.
        output_dir (str): The directory where the image will be saved.
        output_filename (str): The filename for the saved image.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{output_filename}.png")
    save_image(image_tensor, filename)

def save_images(image_array, base_folder, base_filename):
    """
    Saves a list of images to the specified folder with incremental filenames.
    Args:
        image_array (list): List of torch.Tensor images to save.
        base_folder (str): Destination folder for the images.
        base_filename (str): Base filename for the images.
    """
    os.makedirs(base_folder, exist_ok=True)
    for i, img in enumerate(image_array):
        filename = os.path.join(base_folder, f"{base_filename}_{i}.png")
        save_image(img, filename)

def inverse_normalize():
    """
    Returns an inverse normalization transformation.
    Returns:
        transforms.Normalize: Transformation to apply inverse normalization.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

