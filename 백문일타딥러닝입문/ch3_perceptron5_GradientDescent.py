## 경사하강법 실험해보기
import numpy as np
import matplotlib.pyplot as plt

# 여러 가지 학습률을 사용하여 값의 변화를 관찰해보도록 합니다.
lr_list = [0.001, 0.1, 0.3, 0.9]

def get_derivative(lr):
    
    w_old = 2
    derivative = [w_old]

    y = [w_old ** 2] # 손실 함수를 y= x^2 로 정의합니다.

    for i in range(1, 10):
        # 먼저 해당 위치에서 미분값을 구합니다.
        dev_value = w_old * 2

        # 위의 값을 이용하여 가중치를 업데이트합니다.
        w_new = w_old - lr * dev_value
        w_old = w_new

        derivative.append(w_old) # 업데이트 된 가중치를 저장합니다.
        y.append(w_old ** 2) # 업데이트 된 가중치의 손실값을 저장합니다.
        
    return derivative, y

x = np.linspace(-2, 2, 50) # -2 ~ 2의 범위를 50구간으로 나눈 배열을 반환합니다.
x_square = [i ** 2 for i in x]

fig = plt.figure(figsize = (12, 7))

for i, lr in enumerate(lr_list):
    derivative, y = get_derivative(lr)
    ax = fig.add_subplot(2, 2, i + 1)
    ax.scatter(derivative, y, color = 'red')
    ax.plot(x, x_square)
    ax.title.set_text('lr = ' + str(lr))

plt.show()
