import os

import cv2
import numpy as np

# Haar Cascade 얼굴 탐지기 로드
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 이미지가 저장된 폴더 경로
input_folder = "C:/Users/1201q/python/dataset/sindong"
output_folder = "C:/Users/1201q/python/dataset/sindong/face"

# output_folder가 존재하지 않으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 모든 파일에 대해 작업 수행
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)

        # 이미지 읽기
        with open(image_path, "rb") as file:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            continue

        # 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 얼굴 탐지
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for i, (x, y, w, h) in enumerate(faces):
            # 얼굴 영역 추출
            face_image = image[y : y + h, x : x + w]

            # 얼굴 이미지 저장
            output_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, face_image)

print("얼굴 추출 완료!")
