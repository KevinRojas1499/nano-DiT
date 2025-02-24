import os
import torch
import random
import zipfile
import PIL.Image as Image
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def get_files_recursively(path):
    files = []
    with os.scandir(path) as it:
        for e in it:
            if e.is_file():
                files.append(os.path.join(path, e.name))
            elif e.is_dir():
                files.extend(get_files_recursively(os.path.join(path, e.name)))
    return files

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

class FolderOrZipDataset(Dataset):
    def __init__(self, path, size=512, transform=None):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]) if transform is None else transform
        
        # Get list of image files from zip
        self.is_zip = False
        if path.endswith('.zip'):
            self.is_zip = True
            with zipfile.ZipFile(path, 'r') as zip_ref:
                self.files = [f for f in zip_ref.namelist() 
                                if f.lower().endswith(IMAGE_EXTENSIONS)]
        else:
            files = get_files_recursively(path)
            self.files = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
            
        top_files = [f.split('/')[0] for f in self.files]
        unique_labels = enumerate(sorted(set(top_files)))
        labels_map = {lab: i for i, lab in unique_labels}
        self.labels = [labels_map[f] for f in top_files]

    
    def __len__(self):
        return len(self.files)
    
    def open_zip_image(self, idx):
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            img_path = self.files[idx]
            with zip_ref.open(img_path) as file:
                img = Image.open(file).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img

    def open_image(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.is_zip:
            img = self.open_zip_image(idx)
        else:
            img = self.open_image(idx)
        return img, label, idx

class LatentDataset(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, index_path, scale_factor=0.18215, vae=None, **kwargs):
        super(LatentDataset, self).__init__()
        self.vae = vae 
        self.scale_factor = scale_factor

        with open(index_path, 'r') as f:
            self.index = json.load(f)
            self.tot = len(self.index)
        
    def __getitem__(self, idx):
        item = self.index[f'{idx}']
        label = item['cond']
        paths = item['img']
        if isinstance(paths, list):
            path = random.choice(paths)
        else:
            path = paths
        image = np.load(path, mmap_mode='r').copy()
        mean, std = np.split(image, 2, axis=0)
        mean = torch.from_numpy(mean)
        std = torch.from_numpy(std)
        image = mean + std * torch.randn_like(mean)
        image = image * self.scale_factor

        return image, label

    def __len__(self):
        return self.tot

    @torch.no_grad() 
    def decode(self, encoded_images):
        assert self.vae is not None, 'The vae needs to be passed to the dataset for decoding'
        encoded_images = encoded_images / self.scale_factor
        img = self.vae.to(encoded_images.device).decode(encoded_images)
        img = (img + 1)/2
        img = img.clamp(0,1)
        return img 
    
def get_dataset(name, index_path, vae=None):
    if name == 'imagenet':
        return LatentDataset(index_path=index_path, vae=vae)