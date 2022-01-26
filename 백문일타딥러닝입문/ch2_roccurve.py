
#1. Imbalanced Data

import numpy as np
import matplotlib.pyplot as plt

# 임의의 데이터를 생성합니다.
data = [np.random.randint(0, 10) for _ in range(101)]

# 클래스 6에 해당하는 데이터를 30개 더 추가합니다.
[data.append(6) for _ in range(30)]
    

n, bis, patches = plt.hist(data); plt.title('Imbalanced Data', fontsize = 20)
patches[6].set_fc('orange')
plt.show()


#2. ROC CURVE

# [0] : 정상 햄버거 
# [1] : 유통기한 지난 햄버거
y_true = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0]

# 각 클래스에 대한 확률은 임의로 지정합니다.
A_proba = [0.6, 0.7, 0.7, 0.8, 0.9, 0.7, 0.85, 0.7, 0.65, 0.75]
B_proba = [0.05, 0.05, 0.1, 0.3, 0.6, 0.3, 0.4, 0.5, 0.2, 0.1]

# 완벽한 모델
C_pred  = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
C_proba = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0]

from sklearn.metrics import roc_curve, auc

#ROC 곡선을 그리기 위한 값과 AUC 값을 변수에 저장합니다.
fpr_A, tpr_A, thr_A = roc_curve(y_true, A_proba)
fpr_B, tpr_B, thr_B = roc_curve(y_true, B_proba)
fpr_C, tpr_C, thr_C = roc_curve(y_true, C_proba)

auc_A = auc(fpr_A, tpr_A)
auc_B = auc(fpr_B, tpr_B)
auc_C = auc(fpr_C, tpr_C)

import matplotlib.pyplot as plt

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve', fontsize = 20)

plt.plot(fpr_A, tpr_A, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % auc_A)
plt.plot(fpr_B, tpr_B, color='blue',
         lw=2, label='ROC curve (area = %0.2f)' % auc_B)
plt.plot(fpr_C, tpr_C, color='green',
         lw=2, label='ROC curve (area = %0.2f)' % auc_C)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.legend(loc="lower right") # 범례를 추가합니다.
plt.show()

