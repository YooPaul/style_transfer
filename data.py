import torch
import os
from PIL import Image

class CocoDetection(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]


    def __getitem__(self, index):

        path = os.path.join(self.root, self.files[index])

        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img


    def __len__(self):
        return len(self.files)
