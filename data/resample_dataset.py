import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

from utils import find_files, print_metadata

"""
resample_dataset.py
오디오(urbansound8k)의 텐서 크기를 일정하게 맞추는 파일

input : --dataset_path, --resampled_path, --target_sr
학습 데이터 경로, 처리된 데이터 저장 경로, 샘플링 속도

1. 학습 데이터 경로에서 .mp3, .wav, .flac 파일을 검색
2. resampled_path에 데이터 경로 생성
3. 오디오 전처리(process_file)
    1) 오디오를 모노로 전환
    2) 텐서(오디오)의 크기를 줄임
    3) 샘플링 속도를 조절하여 저장
    * 실제 오디오 길이가 줄어드는 것은 아니고, 품질이 조금 떨어짐.
"""

def process_file(input_filename, output_filename, target_sr):
    try:
        """
        샘플링 속도 : 초당 샘플의 수. 영상으로 따지면 화질 정도로, 많을수록 품질이 좋지만 데이터의 크기가 커진다.
        스테레오 : 주로 왼쪽/오른쪽 채널로 구성되어 입체적인 소리를 가짐
        모노 : 단일 체널로 구성되어 단순한 소리를 가짐        
        """
        """
        UrbanSound8k의 첫 번째 데이터(7061-6-0-0)의 audio, ariginal_sr
        audio.shape = torch.Size([2, 99225])
        sample_sr = 44100
        >>> 오디오 크기가 99225이고, 샘플링 속도가 44100이므로 오디오 길이는 약 2초가 된다(99225/44100).
            또 첫 번째 차원 크기가 2이므로 스테레오를 의미한다.(1이면 모노)
        """
        # audio : 오디오 데이터, original_sr : 원본 오디오 샘플링 속도
        audio, original_sr = torchaudio.load(input_filename)

        # 스테레오를 모노로 변환
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        # 정규화 : 오디오의 크기를 0~1 또는 -1~1로 변환
        # 변환 과정에서 진폭만을 변경하기에 피치나 톤은 변하지 않음. 즉, 볼륨만 조정함
        audio /= audio.max()

        # 오디오의 샘플링 속도를 변환.
        # Resample : 오디오의 길이(텐서의 두 번째 차원의 크기)를 샘플링 속도에 맞춤.
        # save : 샘플링 속도를 target_sr로 저장
        # >>> 오디오 텐서의 크기와 샘플링 속도를 모두 줄이는데, 비율은 그대로 유지한다. 즉, 단순히 크기만 맞추고 품질만 조절했음
        audio = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)(audio)
        torchaudio.save(output_filename, audio, target_sr)

    # 진행 중 생긴 오류를 출력
    except Exception as e:
        # print_metadata는 데이터의 세부 내용을 출력. 디버깅 용도인듯
        """
        sample_rate: 오디오 파일의 샘플링 속도(샘플링 주파수)입니다.
        num_channels: 오디오 파일의 채널 수입니다. 단일 채널 오디오인 경우 1이고, 스테레오 오디오인 경우 2입니다.
        num_frames: 오디오 파일의 총 프레임 수입니다.
        bits_per_sample: 각 샘플당 비트 수입니다.
        encoding: 오디오 데이터의 인코딩 방식입니다.
        """
        print_metadata(input_filename)
        print_metadata(output_filename)
        raise e


if __name__ == "__main__":
    # 커맨드 라인(CMD)에서 실행하기 위한 내용. dataset_path를 불러오려면 args.dataset_path를 사용한다.
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--resampled_path", required=True)
    ap.add_argument("--target_sr", default=16000, type=int)
    args = ap.parse_args()

    # resampled_path(처리된 데이터 저장할 경로)가 없으면 생성
    if not os.path.isdir(args.resampled_path):
        os.makedirs(args.resampled_path)

    # find_files : 해당 경로 내의 extensions 확장자를 갖는 파일 탐색. 
    # 참고로 extensions의 default값은 ".mp3", ".wav", ".flac"이다.
    files = list(find_files(args.dataset_path))

    # tqdm은 사실 있으나 마나 실행에는 큰 차이가 없으나, 프로그래스 바를 보여준다.
    for f_in in tqdm(files):
        # f_out : 처리된 데이터 저장 경로. 기존과 이름은 같지만 폴더를 바꿈
        f_out = f_in.replace(args.dataset_path, args.resampled_path)

        # os.path.split : "/path/to/file.txt"와 같은 경로를 인자로 주면 ("/path/to", "file.txt")와 같은 튜플을 반환합니다.
        dir_out, _ = os.path.split(f_out)
        # 경로 없으면 만듦
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)

        # 위의 함수 실행
        process_file(f_in, f_out, target_sr=args.target_sr)
