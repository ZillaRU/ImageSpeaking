import numpy as np
from PIL import Image
from packaging import version
import torch


parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_less_than_1_8 = parsed_torch_version_base < version.parse("1.8.0")

def load_tag_list(tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    if is_torch_less_than_1_8:
        return tensor1 // tensor2
    else:
        return torch.div(tensor1, tensor2, rounding_mode="floor")

def preprocess(image_path, image_size=(384,384) , mean = np.array([0.485, 0.456, 0.406]), std = np.array([0.229, 0.224, 0.225])):
    image = Image.open(image_path)

    if image is None:
        raise ValueError(f"Image at path {image_path} is NOT EXISTS")
    image_pil = image.convert("RGB")

    image_pil_resize = image_pil.resize((image_size))

    # Convert the PIL image to a NumPy array
    image_np = np.array(image_pil_resize)
    image_np = (image_np / 255.0 - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))

    # Add a batch dimension to match the shape (1, C, H, W)
    image_np = np.expand_dims(image_np, axis=0)

    return image_np.astype(np.float32)
