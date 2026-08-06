'''
ch8.py

P.332

빅분기 실기 ch4

'''
# 1. 라이브러리
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 2. 데이터 불러오기
train = pd.read_csv('flight_train.csv')
test = pd.read_csv('flight_test.csv')

# print(train.head())
# print(test.head())

# 3. 데이터 탐색
# print(train.shape, test.shape)  #(10505, 11) (4502, 10)

# print(train.info())
# print(test.info())

# 결측치 확인
# print(train.isnull().sum())
# print(test.isnull().sum())

# print(train['price'].describe())

# 인코딩 대상 컬럼
# print(train.columns[train.dtypes == object])

# 인코딩 대상
cols = ['airline', 'source_city', 'departure_time', 'stops',
       'arrival_time', 'destination_city', 'class']

# 타겟 분리
target = train.pop('price')

# 합치기
df= pd.concat([train, test])
# print(df.shape) # (15007, 10)

# 레이블 인코딩
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# print(df.head())

# 분리
train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):].copy()

# print(train.shape, test.shape)  # (10505, 10) (4502, 10)

# 4. 전처리

train = train.drop('flight', axis = 1)
test = test.drop('flight', axis = 1)
# print(train.shape, test.shape)  # (10505, 9) (4502, 9)

# 원-핫 인코딩
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# 5.데이터 나누기
X_tr, X_val, y_tr, y_val = train_test_split(train, target, test_size = 0.2, random_state = 0)

# print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape) # (8404, 9) (2101, 9) (8404,) (2101,)

# 6. 머신러닝 학습
# 랜덤포레스트 회귀
rf = RandomForestRegressor(random_state = 0)
rf.fit(X_tr, y_tr)  # 학습
y_pred = rf.predict(X_val)
result = root_mean_squared_error(y_val, y_pred)
# print(result)   # 4431.290931404485

pred = rf.predict(test)
submit = pd.DataFrame({'pred':pred})
submit.to_csv('result.csv')

print(pd.read_csv('result.csv').head())