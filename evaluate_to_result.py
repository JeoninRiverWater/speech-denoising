import torch
import torchaudio
import torch.nn.functional as F

# 원본 오디오 파일과 예측 오디오 파일의 경로
original_audio_path = "C:/Users/User/OneDrive/바탕 화면/coding/AudioProcessing/project/result/UNet_new_eval/00/clean.wav"
predicted_audio_path = "C:/Users/User/OneDrive/바탕 화면/coding/AudioProcessing/project/result/evaluation_new_data/00/clean.wav"

# 오디오 파일 로드
waveform_original, sample_rate_original = torchaudio.load(original_audio_path)
waveform_predicted, sample_rate_predicted = torchaudio.load(predicted_audio_path)

# 샘플레이트가 다를 경우 이를 맞춰줘야 하지만, 여기서는 같다고 가정합니다.
# 만약 다르다면 resample을 해야 합니다.

# 손실 함수 적용
loss = F.mse_loss(waveform_predicted, waveform_original)

print(f'MSE Loss: {loss.item()}')
