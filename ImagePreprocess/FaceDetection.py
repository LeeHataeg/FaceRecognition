# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:17:37 2024

@author: gkxor
"""

import cv2
import os

def preprocess_image(image_path, output_path, size=(256, 256)):
    # OpenCV의 얼굴 검출기인 Haar Cascade 사용
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return
    
    # 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴이 검출되면 처리
    for (x, y, w, h) in faces:
        face = gray_image[y:y+h, x:x+w]
        # 256x256 크기로 조정
        resized_face = cv2.resize(face, size)
        # 전처리된 이미지를 파일로 저장
        cv2.imwrite(output_path, resized_face)
        break  # 첫 번째 얼굴만 사용 (필요에 따라 조정 가능)

# 연예인 이미지들이 저장된 폴더와 전처리된 이미지를 저장할 폴더 경로 설정
input_folder = r'C:\Users\gkxor\Documents\GitHub\FaceRecognition\WebCroling\Code\Data\M\배우 이병헌'
output_folder = 'Output/'

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 입력 폴더의 모든 이미지 파일 전처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        preprocess_image(input_path, output_path)