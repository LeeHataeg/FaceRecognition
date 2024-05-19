# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:21:22 2024

@author: gkxor
"""

import os
import time
from selenium import webdriver
from urllib.parse import quote_plus
import requests

# 연예인 이름과 저장할 디렉토리 지정
celebrity_name = "bts 진"
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

# 이미지 URL 추출 및 다운로드
from bs4 import BeautifulSoup

# 웹 페이지 소스 가져오기
html_source = driver.page_source

# BeautifulSoup으로 파싱
soup = BeautifulSoup(html_source, 'html.parser')

# 이미지 URL 추출 및 다운로드
downloaded_count = 0
for i, img in enumerate(soup.find_all('img', class_='rg_i')):
    if downloaded_count >= 150:
        break
    img_url = img.get('src')
    # 이미지 URL이 있는 경우에만 다운로드
    if img_url:
        try:
            img_data = requests.get(img_url).content
            # 중복된 이미지인지 확인
            file_path = os.path.join(save_directory, f"{celebrity_name}_{downloaded_count}.jpg")
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                print(f"다운로드 완료: {img_url}")
                downloaded_count += 1
        except Exception as e:
            print(f"다운로드 실패: {img_url}, {e}")

# 웹드라이버 종료
# driver.quit()