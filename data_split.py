import os
import random
import shutil
from tqdm import tqdm
from data.utils import find_files

def split_train_test(directory, output_directory, train_ratio=0.8, delete_original=False):
    files = list(find_files(directory))
    print(f"Total files found: {len(files)}")
    # 파일을 무작위로 섞음
    random.shuffle(files)
    # 훈련 및 테스트 세트의 분할 인덱스 계산
    split_index = int(train_ratio * len(files))
    train_files = files[:split_index]
    test_files = files[split_index:]

    if os.path.isdir(output_directory) : 
        shutil.rmtree(output_directory)

    # train 디렉토리 및 test 디렉토리 생성
    train_dir = os.path.join(output_directory, 'train')
    test_dir = os.path.join(output_directory, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 훈련 파일 복사
    print("Copying train files...")
    for file in tqdm(train_files, desc="Train Files"):
        destination = os.path.join(train_dir, os.path.basename(file))
        if not os.path.exists(destination):
            shutil.copy(file, destination)
    
    # 테스트 파일 복사
    print("Copying test files...")
    for file in tqdm(test_files, desc="Test Files"):
        destination = os.path.join(test_dir, os.path.basename(file))
        if not os.path.exists(destination):
            shutil.copy(file, destination)

    # 기존 파일 삭제
    if delete_original:
        print("Deleting original files...")
        for file in tqdm(files, desc="Deleting Files"):
            os.remove(file)

# 사용 예시
if __name__ == '__main__' : 
    directory = 'AudioProcessing/speech-denoising/datasets/LibriSpeech_16kHz_4s/train-clean-100'
    output_directory = 'AudioProcessing/speech-denoising/datasets/LibriSpeech_16kHz_4s/train-clean-100-eval'
    split_train_test(directory, output_directory, 0.01, False)
