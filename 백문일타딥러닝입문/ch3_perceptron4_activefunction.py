## 여러가지 활성화 함수

import numpy as np
import matplotlib.pyplot as plt
import math

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 하이퍼볼릭탄젠트 함수
def tanh(x):
    return list(map(lambda x : math.tanh(x), x))

# relu 함수
def relu(x):
    result = []
    for ele in x:
        if(ele <= 0):
            result.append(0)
        else:
            result.append(ele)
            
    return result
# 시그모이드 함수 그려보기
x = np.linspace(-4, 4, 100)
sig = sigmoid(x)

plt.plot(x, sig); plt.title('sigmoid', fontsize = 20)
plt.show()  

## 나머지 그래프도 그려보세요

x = np.linspace(-4, 4, 100)
tan_h = tanh(x)   
relu_d = relu(x)  

plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
plt.plot(x, tan_h); plt.title('tanh', fontsize = 20)
plt.subplot(1, 2, 2)
plt.plot(x, relu_d); plt.title('relu', fontsize = 20)

plt.show()  


## 경사하강법 그리기 2차 함수
x = np.linspace(-2, 2, 50)
x_square = [i ** 2 for i in x]

x_2 = np.linspace(-2, 2, 10)
dev_x = [i ** 2 for i in x_2]

plt.title('x^2 function', fontsize = 20)
plt.plot(x, x_square)
fig = plt.scatter(x_2, dev_x, color = 'red')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()


## 4차 함수
x = np.linspace(-10, 10, 300)
four_func = [(i)*(i - 1)*(i + 1)*(i + 3) for i in x]

fig = plt.figure(figsize = (7, 7))
plt.title('x^4 function', fontsize = 20)
plt.plot(x, four_func)
plt.xlim(-10, 5); plt.ylim(-10, 10)
frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()


## 학습률이 작은 경우
x = np.linspace(-2, 2, 50)
x_square = [i ** 2 for i in x]

x_2 = np.linspace(-2, -1, 25)
dev_x = [i ** 2 for i in x_2]

plt.plot(x, x_square)
fig = plt.scatter(x_2, dev_x, color = 'red')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

## 학습률이 큰 경우
x = np.linspace(-2, 2, 50)
x_square = [i ** 2 for i in x]

x_2_a = np.linspace(-2, -1, 3)
x_2_b = np.linspace(0.8, 1.8, 3)
x_2 = np.concatenate((x_2_a, x_2_b))
dev_x = [i ** 2 for i in x_2]

a_list = []; b_list = []
for a, b in zip(x_2_a, x_2_b[::-1]):
    a_list.append(a)
    a_list.append(b)
    b_list.append(a ** 2)
    b_list.append(b ** 2)

plt.plot(x, x_square) # 함수를 그려주고,
fig = plt.scatter(x_2, dev_x, color = 'red') # 점을 그려주고,
plt.plot(a_list, b_list) # 선을 그립니다.
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()
