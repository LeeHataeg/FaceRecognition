# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:29:19 2024

@author: gkxor
"""

import os
import json
import numpy as np
from keras_facenet import FaceNet
import tensorflow as tf
from scipy.spatial.distance import cosine
import cv2

# 경고 억제
tf.get_logger().setLevel('ERROR')

# FaceNet 모델 로드
embedder = FaceNet()

# 얼굴 임베딩 비교 함수
def compare_embeddings(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# 그룹의 임베딩 정보 로드 함수
def load_group_embeddings(group_file):
    with open(group_file, 'r', encoding='utf-8') as f:
        group_embeddings = json.load(f)
    return group_embeddings

# 이미지의 임베딩 추출 함수 (수정된 부분)
def extract_image_embedding(image_path, embedder):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 얼굴 검출 및 임베딩 추출
    embeddings = embedder.embeddings([img])
    
    # 2차원 배열을 1차원으로 변환하여 반환
    return embeddings[0].flatten()

# 주어진 이미지와 각 그룹의 유사도 계산
def analyze_similarity(image_path, grouped_embeddings):
    image_embedding = extract_image_embedding(image_path, embedder)
    
    similarity_results = []
    for group_name, group_file in grouped_embeddings.items():
        group_embeddings = load_group_embeddings(group_file)
        group_similarity = 0
        
        for _, embedding in group_embeddings.items():
            similarity = compare_embeddings(image_embedding, embedding)
            group_similarity += similarity
        
        group_similarity /= len(group_embeddings)  # 평균 유사도 계산
        similarity_results.append((group_name, group_similarity))
    
    # 유사도가 높은 순으로 정렬
    similarity_results.sort(key=lambda x: x[1], reverse=True)
    return similarity_results

# 그룹화된 임베딩 정보 로드
grouped_embeddings = {}
output_directory = "C:/Users/gkxor/Documents/GitHub/FaceRecognition/ImagePreprocess/files/"
for root, _, files in os.walk(output_directory):
    for file in files:
        if file.endswith(".json"):
            group_name = os.path.splitext(file)[0]
            grouped_embeddings[group_name] = os.path.join(root, file)

# 주어진 이미지와 각 그룹의 유사도 분석
image_path = "test.jpeg"
similarity_results = analyze_similarity(image_path, grouped_embeddings)

# 결과 출력
for group_name, similarity in similarity_results:
    print(f"{group_name} ({similarity:.4f})")