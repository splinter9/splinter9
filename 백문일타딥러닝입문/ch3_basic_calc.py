## 텐서의 차원과 기본 연산

import tensorflow as tf
import numpy as np

print(f'현재 텐서플로우 버전은? {tf.__version__}')

a = tf.constant(2) # 텐서를 선언합니다. 
b = tf.constant([1, 2])
c = tf.constant([[1, 2], [3, 4]])

print(tf.rank(a)) # 텐서의 랭크를 계산합니다.
print(tf.rank(b))
print(tf.rank(c))

a = tf.constant(3)
b = tf.constant(2)

print(tf.add(a, b)) # 더하기
print(tf.subtract(a, b)) # 빼기
print(tf.multiply(a, b).numpy()) # 곱하기
print(tf.divide(a, b).numpy()) # 나누기

## 텐서에서 넘파이로, 넘파이에서 텐서로
c = tf.add(a, b).numpy() # a와 b를 더한 후 NumPy 배열 형태로 변환합니다.
c_square = np.square(c, dtype = np.float32) # NumPy 모듈에 존재하는 square 함수를 적용합니다.
c_tensor = tf.convert_to_tensor(c_square) # 다시 텐서로 변환해줍니다.

# 넘파이 배열과 텐서 각각을 확인하기 위해 출력합니다.
print(f'numpy array : {c}, applying square with numpy : {c_square}, convert_to_tensor : {c_tensor}')
# from tensorflow.math import sin, cos, tanh
# from tensorflow.linalg import diag, svd, matrix_transpose

import tensorflow as tf

@tf.function
def square_pos(x):
    if x > 0:
        x = x * x
    else:
        x = x * -1
    return x

print(square_pos(tf.constant(2)))
print(square_pos.__class__)

def square_pos(x):
    if x > 0:
        x = x * x
    else:
        x = x * -1
    return x

print(square_pos(tf.constant(2)))
print(square_pos.__class__)

