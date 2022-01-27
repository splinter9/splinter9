## MNIST 데이터셋 다운로드
from tensorflow.keras.datasets.mnist import load_data

# 텐서플로우 저장소에서 데이터를 다운받습니다.
(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')

## 데이터 형태 확인하기

# 훈련 데이터
print(x_train.shape, y_train.shape)
print(y_train)

print('\n')

# 테스트 데이터
print(x_test.shape, y_test.shape)
print(y_test)

## 데이터 그리기
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(777)

# 0~59999의 범위에서 무작위로 3개의 정수를 뽑습니다.
sample_size = 3
random_idx = np.random.randint(60000, size = sample_size)

for idx in random_idx:
    img = x_train[idx, :]
    label = y_train[idx]
    plt.figure()
    plt.imshow(img)
    plt.title('%d-th data, label is %d' % (idx,label), fontsize = 15)

plt.show()
    
