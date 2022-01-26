import tensorflow as tf

print(tf.__version__)


## OR 게이트 구현해보기
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse

tf.random.set_seed(777)

# 데이터 준비하기
data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
label = np.array([[0], [1], [1], [1]])

# 모델 구성하기
model = Sequential()
model.add(Dense(1, input_shape = (2, ), activation = 'linear')) # 단층 퍼셉트론을 구성합니다

# 모델 준비하기
model.compile(optimizer = SGD(), loss = mse, metrics = ['acc']) # list 형태로 평가지표를 전달합니다

# 학습시키기
model.fit(data, label, epochs = 500)


## 모델 가중치 확인하기
model.get_weights()

