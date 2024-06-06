# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:43:23 2024

@author: gkxor
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import face_recognition

# 학습된 모델 경로
model_path = 'C:/Users/gkxor/Documents/GitHub/FaceRecognition/model.pkl'

# 모델 불러오기
model = joblib.load(model_path)

# 테스트할 이미지 경로
test_image_path = 'C:/Users/gkxor/Documents/GitHub/FaceRecognition/test.jpg'

# 이미지에서 얼굴 인식하여 임베딩 생성
image = face_recognition.load_image_file(test_image_path)
face_encodings = face_recognition.face_encodings(image)

if len(face_encodings) > 0:
    # 입력 벡터 일반화
    in_encoder = Normalizer(norm='l2')
    face_encodings = in_encoder.transform(face_encodings)
    
    # 모델을 사용하여 얼굴 예측
    predictions = model.predict(face_encodings)
    probability = model.predict_proba(face_encodings)

    # 암호화된 레이블을 디코딩하여 예측된 이름 얻기
    out_encoder = LabelEncoder()
    out_encoder.classes_ = np.load('C:/Users/gkxor/Documents/GitHub/FaceRecognition/classes.npy')
    predict_names = out_encoder.inverse_transform(predictions)
    
    # 예측된 이름 출력
    print('Predicted Name:', predict_names[0])
    print('Probability:', np.max(probability) * 100, '%')
else:
    print('No face detected in the image.')