import numpy as np
from scipy.io.wavfile import write

# WAV 파일에 저장할 총 시간
total_time = 10  # 10초

# 총 시간에 대한 샘플 수
sample_rate = 44100  # 샘플링 레이트
total_samples = total_time * sample_rate

# 각 주파수에 대한 배열 생성
frequencies = [440, 550, 660, 770]  # 예시 주파수

# WAV 파일에 저장할 소리 데이터 초기화
audio_data = np.zeros(total_samples)

# 각 주파수에 대한 소리 데이터 생성 및 더하기
for frequency in frequencies:
    for i in range(int(44100/100)) : 
        time = np.linspace(0, total_time, total_samples)
        sound_wave = 0.3 * np.sin(2 * np.pi * frequency * time)
        audio_data += sound_wave

# 진폭 조정 및 int16 형식으로 변환
audio_data = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)

# WAV 파일로 저장
write("output.wav", sample_rate, audio_data)
