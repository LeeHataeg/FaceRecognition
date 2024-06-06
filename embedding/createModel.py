# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:43:15 2024

@author: gkxor
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle

datasets_path = os.path.join(os.getcwd(), "train_dataset/female/")

def load_data(dataset_path):
    folders = os.listdir(dataset_path)
    X = []
    y = []

    for folder in folders:
        class_path = os.path.join(dataset_path, folder)
        if os.path.isdir(class_path):
            embeddings_path = os.path.join(class_path, "embeddings.npy")
            if os.path.exists(embeddings_path):
                embeddings = np.load(embeddings_path)
                labels = [folder] * embeddings.shape[0]
                X.extend(embeddings)
                y.extend(labels)
    return np.array(X), np.array(y)


# 데이터 로드
X_train, y_train = load_data(datasets_path)

print(X_train)
print("-------------------------------------------------------------------")
# 입력 벡터 일반화
in_encoder = Normalizer(norm='l2')

X_train = in_encoder.transform(X_train)

print(X_train)

# 목표 레이블 암호화
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)

# 모델 적합
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)



# classes.npy 파일 저장
np.save('C:/Users/gkxor/Documents/GitHub/FaceRecognition/female_classes.npy', out_encoder.classes_)

# 모델 저장
model_path = 'C:/Users/gkxor/Documents/GitHub/FaceRecognition/female_model.pkl'
joblib.dump(model, model_path)