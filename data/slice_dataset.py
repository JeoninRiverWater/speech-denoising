import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

from utils import find_files, print_metadata

"""
slice_dataset.py
오디오의 길이를 일정하게 맞추는 파일

input : --dataset_path, --sliced_path, --length_seconds --pad_last
데이터 경로, 저장 경로, 오디오 길이, 패딩 여부

1. 학습 데이터 경로에서 .mp3, .wav, .flac 파일을 검색
2. resampled_path에 데이터 경로 생성
3. 오디오 전처리(process_file)
    1) 오디오 파일의 경로와 확장자를 저장
    2) 세그먼트의 수를 파악
    3) 세그먼트를 맞추기 위해 제로패딩(--pad_last가 False면 없음)
    4) 오디오를 길이에 맞춰 자르고 저장
"""

def process_file(input_filename, output_dir, length_seconds=4, pad_last=True):
    try:
        # os.path.split : ("/path/to", "file.txt")
        base_path, filename = os.path.split(input_filename)
        # os.path.splitext : ('/path/to/file', '.txt')
        name, ext = os.path.splitext(filename)

        audio, sr = torchaudio.load(input_filename)

        # 세그먼트 : 분할, 구간... 즉 오디오를 작은 부분으로 자른 조각
        segment_length = sr * length_seconds
        # 주어진 오디오를 몇 개의 세그먼트가 필요한지를 n_segments에 저장한다.
        """
        예를 들어 주어진 오디오가 10초라면, audio.shape[1]//segment_length는 2일 것이다(10을 4로 나눈 몫). 
        그리고 남은 2초를 처리하기 위한 세그먼트가 필요하므로 1을 더해준다(1 if pad_lat else 0)
        """
        n_segments = (audio.shape[1] // segment_length) + (1 if pad_last else 0)

        # pad는 오디오의 길이를 맞추기 위해 0으로 채울 길이이다.
        # 오디오가 10초라면, 마지막에 2초가 남으며, 이를 채우기 위해 0으로 이루어진 2초간의 데이터를 만들 것이다.
        pad = (n_segments * segment_length) - len(audio)
        if pad > 0:
            audio = torch.cat([audio, torch.zeros([1, pad])], dim=1)

        # Save each segment as {output_dir}/{original_name}_XXXX.{ext}
        for i in range(n_segments):
            # 필요한 부분을 잘라낸다.
            audio_segment = audio[:, i * segment_length : (i + 1) * segment_length]
            # str(i).zfill(4) >>> 만약 i가 1이면 0001, i가 15라면 0015
            # ext는 처음 확장자이다. wav파일이었으면 wav로 저장하는 것.
            segment_name = os.path.join(output_dir, f"{name}_{str(i).zfill(4)}{ext}")
            torchaudio.save(segment_name, audio_segment, sr)

    except Exception as e:
        print_metadata(input_filename)
        # print_metadata(output_filename)
        raise e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--sliced_path", required=True)
    ap.add_argument("--length_seconds", default=4, type=int)
    # action의 store_true는 명령에서 언급하면 True, 그렇지 않으면 False를 반환한다.
    ap.add_argument("--pad_last", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.sliced_path):
        os.makedirs(args.sliced_path)

    files = list(find_files(args.dataset_path))
    for f_in in tqdm(files):
        f_out = f_in.replace(args.dataset_path, args.sliced_path)

        # Make sure the destination directory exists
        dir_out, _ = os.path.split(f_out)
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)

        # Process the audio file
        process_file(f_in, dir_out, length_seconds=args.length_seconds, pad_last=args.pad_last)
