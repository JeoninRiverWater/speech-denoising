import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.models import ConvTasNet
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

from data.utils import find_files, make_path
from evaluate import predict_evaluation_data
from getmodel import get_model
from trainer import Trainer

if __name__ == "__main__":
    """
    run_experiments.py
    파일 역할
    
    input
    --clean_train_path(기본값 : .../train_clean-100)
    --clean_val_path  (기본값 : .../test-clean)
    --noise_train_path(기본값 : .../UrbanSound8k_16khz_4s/train)
    --noise_val_path  (기본값 : .../UrbanSound8k_16khz_4s/test)
    --keep_rate
    --epochs
    --lr
    --gradient_clipping
    --checkpoints_folder(기본값 : checkpoint)
    --evaluations_folder(기본값 : ../PROJECT/EVALUATION)
    --ground_truth_name (기본값 : Ground_truth_mixes_16kHz_4s)
    --gpu
    """
    ap = argparse.ArgumentParser()

    # Datasets
    ap.add_argument(
        "--clean_train_path",
        required=False,
        default=os.path.join("datasets", "LibriSpeech_16kHz_4s", "train-clean-100"),
    )
    
    ap.add_argument(
        "--clean_val_path", required=False, default=os.path.join("AudioProcessing", "speech-denoising", "datasets", "LibriSpeech_16kHz_4s", "test-clean")
    )
    ap.add_argument(
        "--noise_train_path", required=False, default=os.path.join("datasets", "UrbanSound8K_16kHz_4s", "train")
    )
    ap.add_argument(
        "--noise_val_path", required=False, default=os.path.join("datasets", "UrbanSound8K_16kHz_4s", "test")
    )
    ap.add_argument("--keep_rate", default=1.0, type=float)

    # Training params
    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--gradient_clipping", action="store_true")

    # Paths
    ap.add_argument("--checkpoints_folder", required=False, default="checkpoints")
    ap.add_argument("--evaluations_folder", required=False, default=os.path.join("..", "PROJECT", "EVALUATION"))
    ap.add_argument("--ground_truth_name", required=False, default="Ground_truth_mixes_16kHz_4s")

    # GPU setup
    ap.add_argument("--gpu", default="-1")

    args = ap.parse_args()

    # assert : assert 조건, "오류 메시지". 즉, 폴더가 있는지 확인하고 없으면 오류 메시지 출력
    assert os.path.isdir(args.checkpoints_folder), "The specified checkpoints folder does not exist"
    assert os.path.isdir(args.evaluations_folder), "The specified evaluations folder does not exist"
    assert os.path.isdir(
        os.path.join(args.evaluations_folder, args.ground_truth_name)
    ), "The specified ground truth folder does not exist"

    # GPU 설정. 여기서 args.gpu의 기본값이 "-1"이므로 gpu 실행을 비활성화한다.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # cpu와 gpu 중 사용할 것을 출력한다. gpu의 기본값은 -1이기에 보통 cpu가 출력된다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    #
    # Initialize the datasets
    #
    from data import AudioDirectoryDataset, NoiseMixerDataset

    # args.keep_rate의 기본값은 1.0이다.
    """
    AudioDirectoryDataset
    root에서 오디오 파일을 keep_rate 비율만큼 가져온다. 
    여기서는 clean_train_path의 모든 데이터를 clean_dataset.filenames에 리스트로 저장된다.
    """
    train_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_train_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_train_path, keep_rate=args.keep_rate),
        # mode=data_mode,
    )

    val_data = NoiseMixerDataset(
        clean_dataset=AudioDirectoryDataset(root=args.clean_val_path, keep_rate=args.keep_rate),
        noise_dataset=AudioDirectoryDataset(root=args.noise_val_path, keep_rate=args.keep_rate),
        # mode=data_mode,
    )

    # args.epochs의 default값은 10 args.lr의 default값은 1e-4(=0.0001)이다. lr은 주로 학습률을 의미한다.
    # UNet, ConvTasNet, TransUNet은 모두 실제로 존재하는 모델이다.
    """
    UNet은 이미지 세그멘테이션 작업에 주로 사용되는 딥러닝 아키텍처입니다. 이미지의 각 픽셀이 어떤 클래스에 속하는지를 분류하는 작업을 수행합니다. 
    이를 통해 이미지 내의 특정 객체나 구조를 정확하게 식별하고 위치를 파악할 수 있습니다. 

    UNetDNP는 UNet과 같은 구조를 따른다. 거기에 더해 다양한 조건에 동적으로 잡음을 예측하고 처리하는 모듈이 포함된다.

    ConvTasNet은 음성 분리 작업을 위한 효율적이고 성능이 우수한 모델로, 하나의 오디오 신호에서 여러 개의 독립적인 음성 신호를 분리하는 역할을 합니다. 
    # 참고로 ConvTasNet은 따로 구현된 것이 아니고 torchaudio.models.ConvTasNet을 사용했다.

    TransUNet은 Transformer 기반의 인코더와 UNet 기반의 디코더를 이용한다. 마찬가지로 객체의 경계를 감지하고 분할한다.
    """
    experiments = [
        {"model": "UNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 16},
        {"model": "UNetDNP", "epochs": args.epochs, "lr": args.lr, "batch_size": 16},
        {"model": "ConvTasNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 8},
        {"model": "TransUNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 4},
    ]

    for experiment in experiments:

        # Select the model to be used for training
        training_utils_dict = get_model(experiment["model"])

        model = training_utils_dict["model"] # UNet
        loss_fn = training_utils_dict["loss_fn"] # F.mse_loss(평균 제곱 오차 손실 함수)
        loss_mode = training_utils_dict["loss_mode"] # 'min'

        data_mode = training_utils_dict["data_mode"] # 'amplitude'
        train_data.mode = data_mode
        val_data.mode = data_mode

        # data_mode가 'amplitude'이므로 "mse"
        loss_name = "sisdr" if data_mode == "time" else "mse"

        # "UNet_mse_"
        model_name = f"{experiment['model']}_{loss_name}_{experiment['lr']}_{experiment['epochs']}_epochs"
        checkpoint_name = os.path.join(args.checkpoints_folder, f"{model_name}.tar")

        print("-" * 50)
        print("Model:", experiment["model"])
        print("Checkpoint:", checkpoint_name)
        print("Loss:", loss_name)
        print("Epochs", experiment["epochs"])
        print("Batch size:", experiment["batch_size"])
        print("Learning rate:", experiment["lr"])

        # Start training
        model = model.to(device)

        if not os.path.isfile(checkpoint_name):
            # Train an generate the model checkpoint if it does not exit. Otherwise skip and evaluate
            tr = Trainer(train_data, val_data, checkpoint_name=checkpoint_name)
            history = tr.fit(
                model,
                device,
                epochs=experiment["epochs"],
                batch_size=experiment["batch_size"],
                lr=experiment["lr"],
                loss_fn=loss_fn,
                loss_mode=loss_mode,
                gradient_clipping=args.gradient_clipping,
            )

        # Generate the folder for the model predictions
        evaluation_output_directory = os.path.join(args.evaluations_folder, model_name)

        if not os.path.isdir(evaluation_output_directory):
            # Restore from the best checkpoint
            checkpoint = torch.load(checkpoint_name)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
            print(f"Model loaded from checkpoint: {checkpoint_name}")

            make_path(evaluation_output_directory)

            # Get predictions for evaluation
            predict_evaluation_data(
                evaluation_directory=os.path.join(args.evaluations_folder, args.ground_truth_name),
                output_directory=evaluation_output_directory,
                model=model,
                data_mode=data_mode,
                length_seconds=4,
                normalize=True,
            )

            del model
