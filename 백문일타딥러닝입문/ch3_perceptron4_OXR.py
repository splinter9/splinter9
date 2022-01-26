## XOR 게이트 구현해보기 + 다층 퍼셉트론 

#####################################
##############  실패  ################
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse

tf.random.set_seed(777)

# 데이터 준비하기
data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
label = np.array([[0], [1], [1], [0]])

# 모델 구성하기
model = Sequential()
model.add(Dense(1, input_shape = (2, ), activation = 'linear'))

# 모델 준비하기
model.compile(optimizer = SGD(), loss = mse, metrics = ['acc'])

# 학습시키기
model.fit(data, label, epochs = 10000)




#####################################
############# 성공 ##################
import tensorflow as tf
tf.random.set_seed(777)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mse

# 데이터 준비하기
data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
label = np.array([[0], [1], [1], [0]])

# 모델 구성하기
model = Sequential()
model.add(Dense(32, input_shape = (2, ), activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 모델 준비하기
model.compile(optimizer = RMSprop(), loss = mse, metrics = ['acc'])

# 학습시키기
model.fit(data, label, epochs = 100)
model.evaluate(data, label)
result = model.predict(data)
print(result)
