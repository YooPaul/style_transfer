from models import *
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser(description='Image style transfer.')
parser.add_argument('model', type=str,
                    help='path to model.pt')
parser.add_argument('image', type=str,
                    help='path to image file')
parser.add_argument('--output', type=str, default='.',
                    help='output path to save image file')

args = parser.parse_args()
to_tensor = transforms.ToTensor()
to_PIL = transforms.ToPILImage()

def style_image(model, image_file, device):
    image = Image.open(image_file).convert('RGB')
    input = to_tensor(image).unsqueeze(0).to(device)
    out = model(input) * 255 # now values in [0, 255]
    img = to_PIL(torch.round(out.squeeze(0)).to(torch.uint8)).convert('RGB')
    return img


if __name__ == '__main__':
    if not os.path.isfile(args.model):
        print('Model cannot be found.')
        exit()
    elif not os.path.isfile(args.image):
        print('Image file does not exist.')
        exit()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = StyleTransfer().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    output = style_image(model, args.image, device)
    filename, file_extension = os.path.splitext(args.image)
    output.save(os.path.join(args.output, 'output' + file_extension))

