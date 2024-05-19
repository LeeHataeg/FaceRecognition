# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:26:10 2024

@author: gkxor
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 사전 학습된 모델 로드 (예: ResNet)
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  # 마지막 레이어 제거
model.eval()

# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image

# 특징 추출 함수
def extract_features(image_tensor):
    with torch.no_grad():
        features = model(image_tensor.unsqueeze(0))
    return features.numpy().flatten()

# 연예인 이미지 디렉토리
celeb_images_dir = 'Data/'

# 연예인 이미지 로드 및 특징 추출
celeb_features = {}
for filename in os.listdir(celeb_images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(celeb_images_dir, filename)
        image_tensor = preprocess_image(image_path)
        features = extract_features(image_tensor)
        celeb_name = os.path.splitext(filename)[0]
        celeb_features[celeb_name] = features.tolist()

# 특징값 저장
with open('celeb_features.json', 'w') as f:
    json.dump(celeb_features, f)
