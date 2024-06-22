import torchaudio

# 오디오 파일 경로 지정
file_path = "AudioProcessing/7061-6-0-0.wav"

audio, sample_rate = torchaudio.load(file_path)

print(audio.shape)
print(sample_rate)

audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
torchaudio.save("AudioProcessing/7061-6-0-0_resampled.wav", audio, 16000)


resampled_file_path = "AudioProcessing/7061-6-0-0_resampled.wav"

res_audio, res_sample_rate = torchaudio.load(resampled_file_path)

print(res_audio.shape)
print(res_sample_rate)