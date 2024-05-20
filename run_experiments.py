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
    'AudioProcessing\speech-denoising\datasets\UrbanSound8k\UrbanSound8K'
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

    #
    # Set the GPU
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({args.gpu})")

    #
    # Initialize the datasets
    #
    from data import AudioDirectoryDataset, NoiseMixerDataset

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

    #
    # Experiments
    #
    experiments = [
        {"model": "UNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 16},
        {"model": "UNetDNP", "epochs": args.epochs, "lr": args.lr, "batch_size": 16},
        {"model": "ConvTasNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 8},
        {"model": "TransUNet", "epochs": args.epochs, "lr": args.lr, "batch_size": 4},
    ]

    for experiment in experiments:

        # Select the model to be used for training
        training_utils_dict = get_model(experiment["model"])

        model = training_utils_dict["model"]
        loss_fn = training_utils_dict["loss_fn"]
        loss_mode = training_utils_dict["loss_mode"]

        data_mode = training_utils_dict["data_mode"]
        train_data.mode = data_mode
        val_data.mode = data_mode

        loss_name = "sisdr" if data_mode == "time" else "mse"

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
