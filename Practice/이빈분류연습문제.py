# 1. 문제 정의
# 평가 : roc_auc
# target: Outcome
# 최종 파일: result.csv(컬럼 1개 pred,1 확률값)

# 2.라이브러리 불러오기
import pandas as pd

train = pd.read_csv('/data_csv/diabeter_train.csv')
test = pd.read_csv('/data_csv/diabetes_test.csv')
