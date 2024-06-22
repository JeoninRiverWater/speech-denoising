import torch
import torch.nn.functional as F

# 예측값
predictions = torch.tensor([0.9, 1.8, 2.7, 3.6])
# 실제값
targets = torch.tensor([1.0, 2.0, 3.0, 4.0])

# MSE 손실 계산
loss = F.mse_loss(predictions, targets)
print(loss)

# 각각의 값의 차를 제곱한다. 그리고 이의 평균을 계산한다.
# ((0.9-1.0)^2 + (1.8-2.0)^2 + (2.7-3.0)^2 + (3.6-4.0)^2 ) / 4
print(round((pow(0.9-1.0, 2)+pow(1.8-2.0, 2)+pow(2.7-3.0, 2)+pow(3.6-4.0, 2))/4, 3))