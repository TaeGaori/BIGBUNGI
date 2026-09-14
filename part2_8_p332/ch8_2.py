'''
ch8_2.py

P.338

노트북 가격 예측

'''


# 1. 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 2.데이터 불러오기
train = pd.read_csv('laptop_train.csv')
test = pd.read_csv('laptop_test.csv')

# print(train.head())
# print(test.head())


# 3. 데이터 탐색
# print(train.shape, test.shape)  # (91, 10) (39, 9)

# print(train.info())
# print(test.info())

# 결측치 확인
# print(train.isnull().sum())
# print(test.isnull().sum())

# print(train['Price'].describe())

# 인코딩 대상 컬럼
# print(train.columns[train.dtypes == object])

# 타겟 분리
target = train.pop('Price')

cols = ['Brand', 'Model', 'Series', 'Processor', 'Processor_Gen',
       'Hard_Disk_Capacity', 'OS']

# 합치기
df = pd.concat([train, test])
# print(df.shape) # (130, 9)

# 레이블 인코딩
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# print(df.shape) # (130, 9)

# 분리
train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):].copy()

# print(train.shape, test.shape)  # (91, 9) (39, 9)

# 4. 전처리

