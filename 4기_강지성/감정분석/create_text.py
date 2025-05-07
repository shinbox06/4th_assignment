
import pandas as pd
import re
import konlpy
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from tqdm import tqdm
import urllib.request
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-2, 2))
# ratings_train.txt 파일을 탭(tab) 구분자로 읽어서 DataFrame으로 저장
# 전체 데이터 중 처음 10,000개 행(row)만 선택
# 원본 데이터에서 'id' 열과 'document' 열만 선택
# 나머지 열(예: 'label' 같은 것들)은 제거
# 열 이름을 변경
# 여기서는 'document' 열의 이름을 'sentence'로 바꿔서 문장을 담는 열임을 명확하게
df = pd.read_excel('유튜브_영상_댓글_수집_20250427140217.xlsx')
df = df["댓글 내용"].rename("sentence").to_frame()

# Regular expression to remove mentions (@username)
def remove_mentions(text):
    if isinstance(text, str):  # Check if the input is a string
        return re.sub(r'@\S+', '', text)
    else:
        return text

# Apply the function to the 'sentence' column
df['sentence'] = df['sentence'].apply(remove_mentions)

# Remove English sentences
def remove_english(text):
    if isinstance(text, str):
      if re.search('[a-zA-Z]', text):
        return ''
      else: return text
    else:
        return text

df['sentence'] = df['sentence'].apply(remove_english)

# Drop rows with NaN values in the 'sentence' column
df.dropna(subset=['sentence'], inplace=True)

#Remove empty strings
df = df[df['sentence'] != '']

print(df.head())