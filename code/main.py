import base64
from typing import List, Optional

import cv2
import face_recognition
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import JSONResponse
from param import String
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://weareugly.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# uvicorn main:app --reload


def detect_faces(image: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40)
    )
    face_images = [image[y : y + h, x : x + w] for (x, y, w, h) in faces]
    return face_images


def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image)
    image_bytes = buffer.tobytes()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return base64_str


@app.post("/isface")
async def detect_isface(file: UploadFile = File(...)):
    try:
        image = face_recognition.load_image_file(file.file)
        is_contain_face = bool(face_recognition.face_encodings(image))

        return {
            "isFace": is_contain_face,
            "message": (
                "이미지에 얼굴이 있습니다."
                if is_contain_face
                else "이미지에 얼굴이 없습니다."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/extract")
async def detect_extract(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(content={"error": "이미지 파일이 없어요."}, status_code=400)

    face_images = detect_faces(image)

    if not face_images:
        return JSONResponse(
            content={"error": "얼굴을 찾을 수 없어요."}, status_code=400
        )

    base64_faces = [image_to_base64(face) for face in face_images]

    return {"faces": base64_faces}


@app.post("/predict")
async def predict_top_similar_faces(file: UploadFile = File(...)):
    try:
        try:
            male_model = joblib.load("male_face_recognition_model.pkl")
            male_in_encoder = joblib.load("male_in_encoder.pkl")
            male_out_encoder = joblib.load("male_out_encoder.pkl")

            female_model = joblib.load("female_face_recognition_model.pkl")
            female_in_encoder = joblib.load("female_in_encoder.pkl")
            female_out_encoder = joblib.load("female_out_encoder.pkl")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {str(e)}")

        try:
            image = face_recognition.load_image_file(file.file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"이미지 로딩 실패: {str(e)}")

        try:
            width = image.shape[1]
            height = image.shape[0]

            face_encodings = face_recognition.face_encodings(
                image, known_face_locations=[(0, width, height, 0)]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"이미지 인코딩 실패: {str(e)}")

        if face_encodings:
            try:
                embedding = face_encodings[0].reshape(1, -1)
                male_embedding = male_in_encoder.transform(embedding)
                female_embedding = female_in_encoder.transform(embedding)

                male_probabilities = male_model.predict_proba(male_embedding)[0]
                female_probabilities = female_model.predict_proba(female_embedding)[0]

                male_predicted_indices = np.argsort(male_probabilities)[::-1][::]
                female_predicted_indices = np.argsort(female_probabilities)[::-1][::]

                male_predictions = []
                female_predictions = []
                for i, index in enumerate(male_predicted_indices):
                    predict_name = male_out_encoder.inverse_transform([index])[0]
                    probability = male_probabilities[index] * 100
                    male_predictions.append(
                        {
                            "rank": i + 1,
                            "name": predict_name,
                            "probability": probability,
                        }
                    )

                for i, index in enumerate(female_predicted_indices):
                    predict_name = female_out_encoder.inverse_transform([index])[0]
                    probability = female_probabilities[index] * 100
                    female_predictions.append(
                        {
                            "rank": i + 1,
                            "name": predict_name,
                            "probability": probability,
                        }
                    )

                return {
                    "male_predictions": male_predictions,
                    "female_predictions": female_predictions,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="얼굴이 없음.")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
def read_root():
    return {"test": "Hello World"}


def app_function(request):
    return WSGIMiddleware(app)(request)
