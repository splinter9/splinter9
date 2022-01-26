#내적해보기
import tensorflow as tf

x = tf.random.unifom((10, 5)) # uniform 분포에서 해당 크기만큼 난수를 생성합니다.
w = tf.random.unifom((5, 3))
d = tf.matmul(x, w) # (10, 5) * (5, 3)

print(f'x와 w의 벡터 내적의 결과 크기:{d.shape}')

# x와 w의 벡터 내적의 결과 크기:(10, 3)
