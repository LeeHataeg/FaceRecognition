# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:08:10 2024

@author: gkxor
"""
import requests

import os
import time
import base64
from selenium import webdriver
from selenium.webdriver.common.by import By  # By 모듈 import 추가
from urllib.parse import quote_plus

# 연예인 이름과 저장할 디렉토리 지정
celebrity_name = "배우 이병헌"
save_directory = "Data/M/" + celebrity_name

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 구글 이미지 검색 URL 생성
search_query = quote_plus(celebrity_name)
url = f"https://www.google.com/search?q={search_query}&tbm=isch"

# Selenium 웹드라이버 설정
options = webdriver.ChromeOptions()
# options.add_argument('headless')  # 화면 표시하지 않음
options.add_argument('window-size=1920x1080')  # 창 크기 설정
options.add_argument('disable-gpu')  # GPU 사용 안함
driver = webdriver.Chrome(options=options)

# 웹 페이지 로드 및 스크롤 다운
driver.get(url)
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # 페이지 로딩을 기다림
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# 이미지 다운로드
downloaded_count = 0
max_attempts = 3  # 최대 시도 횟수
for i, img in enumerate(driver.find_elements(By.XPATH, '//img[contains(@class,"rg_i")]')):
    if downloaded_count >= 150:
        break
    attempts = 0
    while attempts < max_attempts:
        try:
            # 이미지 URL 가져오기
            img_url = img.get_attribute('src')
            if not img_url:
                img_url = img.get_attribute('data-src')

            # Base64 인코딩 이미지인 경우 처리
            if img_url.startswith('data:image'):
                # Base64 디코딩
                img_data = base64.b64decode(img_url.split(',')[1])
            else:
                # 이미지 다운로드
                img_data = requests.get(img_url).content

            # 이미지 저장
            with open(os.path.join(save_directory, f"{celebrity_name}_{downloaded_count}.jpg"), 'wb') as f:
                f.write(img_data)

            print(f"다운로드 완료: {img_url}")
            downloaded_count += 1
            break  # 이미지 다운로드 성공 시에는 루프 종료
        except Exception as e:
            print(f"다운로드 실패: {img_url}, {e}")
            attempts += 1


# 웹드라이버 종료
# driver.quit()



