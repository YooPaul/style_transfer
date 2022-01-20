import torch
from torchvision import transforms
from PIL import Image
import os

from models import StyleTransfer
from perceptual_loss import PerceptualLoss
from data import CocoDetection

EPOCHS = 2
lr = 1e-3
BATCH_SIZE = 4
MODEL_DATA = "data/model.pt"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    #transforms.ToPILImage(), # applies transpose
    transforms.Resize((256, 256)),
    transforms.ToTensor(), #  normalizes image pixel values to [0, 1]
])


train_dataset = CocoDetection('train2017', preprocess)
train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = StyleTransfer().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
step = 0

if os.path.isfile(MODEL_DATA):
    checkpoint = torch.load(MODEL_DATA)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    print('Previous step:', step)
    print('Previous loss', loss)

    model.train()

style_img = Image.open('starry_night.jpeg').convert('RGB')
style_img = vgg_normalize(preprocess(style_img).unsqueeze(0))
style_img = style_img.expand(torch.Size((BATCH_SIZE,-1, -1, -1))).to(device)
style_img.requires_grad = False # no need to compute gradients for the style image

loss_func = PerceptualLoss(device)

# Hyperparameters
lambda_c = 1.0 # content loss weight
lambda_s = 5.0 # style loss weight

for epoch in range(EPOCHS):
    for idx, x in enumerate(train):
        optim.zero_grad()
        x = x.to(device)

        out = vgg_normalize(model(x))
        perceptual_loss = lambda_c * loss_func.feature_reconstruction_loss(out, vgg_normalize(x)) + lambda_s * loss_func.style_reconstruction_loss(out, style_img)
        perceptual_loss.backward()

        optim.step()

        step += 1
        # Print loss and save model every 30 iterations
        if (step - 1) % 30 == 0:
            print('Step:', step)
            print('Loss:', perceptual_loss.item())
        
            torch.save({'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'loss': perceptual_loss.item(),
                        }, MODEL_DATA)
