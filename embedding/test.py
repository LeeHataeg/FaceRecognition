import os
import shutil
from collections import Counter

import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split

datasets_path = os.path.join(os.getcwd(), "dataset/female")
train_path = os.path.join(os.getcwd(), "train_dataset/female")
test_path = os.path.join(os.getcwd(), "test_dataset/female")
custom_test_path = os.path.join(os.getcwd(), "test")

all_folders = os.listdir(train_path)


# dataset을 일정한 비율로 학습 데이터와 테스트 데이터로 나눕니다.
def split_files(dataset_path, train_path, test_path, test_size=0.2, random_state=42):
    folders = os.listdir(dataset_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for folder in folders:
        class_path = os.path.join(dataset_path, folder)
        if os.path.isdir(class_path):
            images = [
                f
                for f in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, f))
            ]
            train_images, test_images = train_test_split(
                images, test_size=test_size, random_state=random_state
            )

            train_folder_path = os.path.join(train_path, folder)
            test_folder_path = os.path.join(test_path, folder)

            if not os.path.exists(train_folder_path):
                os.makedirs(train_folder_path)
            if not os.path.exists(test_folder_path):
                os.makedirs(test_folder_path)

            for image in train_images:
                src_path = os.path.join(class_path, image)
                dst_path = os.path.join(train_folder_path, image)
                shutil.copyfile(src_path, dst_path)

            for image in test_images:
                src_path = os.path.join(class_path, image)
                dst_path = os.path.join(test_folder_path, image)
                shutil.copyfile(src_path, dst_path)


# 각 폴더에 임베딩 데이터를 생성합니다.
def create_embeddings(path, folders):
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        if os.path.exists(os.path.join(folder_path, "embeddings.npy")):
            print(f"{folder}: 임베딩이 존재합니다.")
            continue

        print(f"{folder}: 임베딩을 시작합니다.")

        face_embeddings = []

        # 모든 이미지들 순회하면서 임베딩
        for filename in os.listdir(folder_path):  # 폴더의 모든 파일 순회
            file_path = os.path.join(folder_path, filename)  # 이미지 경로
            if not file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image = face_recognition.load_image_file(file_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                face_embeddings.append(face_encodings[0])

        face_embeddings = np.asarray(face_embeddings)
        np.save(os.path.join(folder_path, "embeddings.npy"), face_embeddings)
        print(f"{folder}: 임베딩이 완료되었습니다.")


# 임베딩 데이터를 로딩하고 embedding_dict를 반환합니다.
def load_embeddings(path, folders):
    embedding_dict = {}
    for folder in folders:
        embedding_path = os.path.join(path, folder, "embeddings.npy")
        if os.path.exists(embedding_path):
            data = np.load(embedding_path)
            embedding_dict[folder] = data
        else:
            print(f"{folder}: 임베딩 로드 실패")
    print(f"모든 임베딩 로드 완료")
    return embedding_dict


def find_similar_person(embedding_dict, image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        print(f"Error: No face found in image {image_path}")
        return None

    image_embedding = face_encodings[0]

    min_distance = float("inf")
    most_similar_person = None
    dicts = {}

    for person, embeddings in embedding_dict.items():
        person_distance = float("inf")
        for embedding in embeddings:
            distance = cosine(image_embedding, embedding)
            if distance < min_distance:
                min_distance = distance
                most_similar_person = person
            if distance < person_distance:
                person_distance = distance
        dicts[person] = person_distance
    sorted_items = sorted(dicts.items(), key=lambda x: x[1])[:5]
    print(sorted_items)
    return most_similar_person


def start_test(train_path, test_path):
    correct = 0
    incorrect = 0
    embedding = load_embeddings(train_path, all_folders)

    for folder_name in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"{folder_name}: 시작")
            image_count = len(os.listdir(folder_path))
            count = 0
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(test_path, folder_name, filename).replace(
                        "\\", "/"
                    )
                    predict_person = find_similar_person(embedding, image_path)
                    if predict_person == folder_name:
                        count = count + 1
                        correct = correct + 1
                    else:
                        incorrect = incorrect + 1

                        if predict_person == None:
                            print(f"{image_path}는 얼굴을 인식하지 못했어요")
                        else:
                            print(
                                f"{image_path}는 {folder_name}을 {predict_person}로 예측했어요"
                            )
            print(f"{folder_name}: {count}/{image_count}")
    print(
        f"{correct + incorrect}개 중에 {correct}개 / {incorrect}개 / 정답률 : {(correct / (correct + incorrect)) * 100}%"
    )
    plot_results(correct, incorrect)


def plot_results(correct, incorrect):
    categories = ["Correct", "Incorrect", "Total"]
    values = [correct, incorrect, correct + incorrect]

    plt.bar(categories, values, color=["green", "red", "blue"])
    plt.xlabel("Categories")
    plt.ylabel("Number of Images")
    plt.title("Classification Results")
    plt.show()


# start_test(train_path, test_path)

# model = load_embeddings(train_path, all_folders)
# find_similar_person(model, "C:/Users/gkxor/Desktop/rs/test.jpg")
# split_files(datasets_path, train_path, test_path)