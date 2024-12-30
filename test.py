# test the model checkpoint.pth 

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from vision_transformer import DINOHead
from vision_transformer import VisionTransformer
from main import get_args_parser, parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import weights pth file
model
model.load_state_dict(torch.load('checkpoint.pth'))
print(model)
model = model
model.to(device)


video_PATH = 'Walking Tour Wildlife.mp4'
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
cap = cv2.VideoCapture(video_PATH)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = transform(frame)
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    with torch.no_grad():
        outputs = model.forward(frame)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
cap.release()
cv2.destroyAllWindows()
