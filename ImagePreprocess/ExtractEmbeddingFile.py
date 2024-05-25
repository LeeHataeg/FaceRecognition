import os
import json
import cv2
import numpy as np
from keras_facenet import FaceNet
import tensorflow as tf

# 경고 억제
tf.get_logger().setLevel('ERROR')

# FaceNet 모델 로드
embedder = FaceNet()

# 얼굴 임베딩 추출 함수
def extract_face_embeddings(image_path, embedder):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 얼굴 검출 및 임베딩 추출
    embeddings = embedder.embeddings([img])
    
    return embeddings[0].tolist()

# 디렉토리 내 이미지 파일에서 얼굴 임베딩 추출 및 저장
def save_face_embeddings_from_directory(directory, output_directory):
    for root, dirs, files in os.walk(directory):
        group_name = os.path.basename(root)  # 디렉토리 이름을 그룹 이름으로 사용
        group_embeddings = {}
        
        for filename in files:
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(root, filename)
                embeddings = extract_face_embeddings(image_path, embedder)
                if embeddings is not None:
                    relative_path = os.path.relpath(image_path, directory)
                    group_embeddings[relative_path] = embeddings
        
        # JSON 파일로 저장
        output_file = os.path.join(output_directory, f"{group_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(group_embeddings, f, indent=4, ensure_ascii=False)

# 사용 예시
input_directory = "C:/Users/gkxor/Documents/GitHub/FaceRecognition/ImagePreprocess/temp/"
output_directory = "C:/Users/gkxor/Documents/GitHub/FaceRecognition/ImagePreprocess/test/"
save_face_embeddings_from_directory(input_directory, output_directory)