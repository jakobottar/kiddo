# pylint: disable=redefined-outer-name, line-too-long
"""
main training script

last updated jan 2024
"""

import os
import random

import configargparse
import mlflow
import namegenerator
import torch
import yaml
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, models, transforms
from tqdm import tqdm


def train_loop(dataloader, model, optimizer, scheduler, configs):
    """
    model training loop
    trains 1 epoch
    """

    # set model to train mode
    model.train()

    # set up loss function and metric
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=configs.num_classes).to(configs.device)

    num_batches = len(dataloader)
    train_loss = 0.0

    tbar_loader = tqdm(dataloader, desc="train", total=num_batches, dynamic_ncols=True, disable=configs.no_tqdm)

    for images, labels in tbar_loader:
        # move images to GPU if needed
        images, labels = images.to(configs.device), labels.to(configs.device)

        # zero gradients from previous step
        optimizer.zero_grad()

        # compute prediction and loss
        logits = model(images)
        loss = loss_fn(logits, labels)
        train_loss += loss.item()

        # backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # update metrics
        accuracy.update(logits, labels)

    return {
        "train_acc": float(accuracy.compute()),
        "train_loss": train_loss / num_batches,
        "learning_rate": scheduler.get_last_lr()[0],
    }


def val_loop(dataloader, model, configs):
    """
    model validation loop
    """

    # set model to eval mode
    model.eval()

    # set up loss function and metric
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=configs.num_classes).to(configs.device)

    num_batches = len(dataloader)
    val_loss = 0.0

    with torch.no_grad():
        tbar_loader = tqdm(dataloader, desc="val", total=num_batches, dynamic_ncols=True, disable=configs.no_tqdm)

        for images, labels in tbar_loader:
            # move images to GPU if needed
            images, labels = images.to(configs.device), labels.to(configs.device)

            # compute prediction and loss
            logits = model(images)
            val_loss += loss_fn(logits, labels).item()

            # update metrics
            accuracy.update(logits, labels)

    return {"val_acc": float(accuracy.compute()), "val_loss": val_loss / num_batches}


if __name__ == "__main__":
    # parse args/config file
    parser = configargparse.ArgParser(default_config_files=["./config.yml"])
    parser.add_argument("--arch", type=str, default="resnet18", help="model architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file, omit for no checkpoint")
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        default="./config.yaml",
        help="config file location",
    )
    parser.add_argument("-r", "--dataset-root", type=str, default="./data/", help="dataset filepath")
    parser.add_argument("--device", type=str, default="cuda", help="gpu(s) to use")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--name", type=str, default="random", help="run name")
    parser.add_argument("--no-tqdm", action="store_true", help="disable tqdm progress bar")
    parser.add_argument("--root", type=str, default="runs", help="root of folder to save runs in")
    parser.add_argument("-S", "--seed", type=int, default=-1, help="random seed, -1 for random")
    parser.add_argument("--skip-train", action="store_true", help="skip training")
    parser.add_argument("--weight-decay", type=float, default=1e-9, help="optimizer weight decay")
    parser.add_argument("--workers", type=int, default=2, help="dataloader worker threads")
    configs, _ = parser.parse_known_args()

    #########################################
    ## SET UP SEEDS AND PRE-TRAINING FILES ##
    #########################################
    if configs.name == "random":
        configs.name = namegenerator.gen()
    else:
        configs.name = configs.name

    if configs.seed != -1:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        cudnn.deterministic = True

    print(f"Run name: {configs.name}")
    try:
        os.makedirs(f"{configs.root}/{configs.name}", exist_ok=True)
    except FileExistsError as error:
        pass
    configs.root = f"{configs.root}/{configs.name}"

    # save configs object as yaml
    with open(os.path.join(configs.root, "config.yml"), "w", encoding="utf-8") as file:
        yaml.dump(vars(configs), file)

    ####################
    ## SET UP DATASET ##
    ####################
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),  # cifar10 mean and sd
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )

    training_data = datasets.CIFAR10(root=configs.dataset_root, train=True, download=True, transform=train_transform)
    val_data = datasets.CIFAR10(root=configs.dataset_root, train=False, download=True, transform=val_transform)
    configs.num_classes = 10

    print(training_data)
    print(val_data)

    train_dataloader = DataLoader(
        training_data,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=configs.workers,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=configs.workers,
    )

    ##################
    ## SET UP MODEL ##
    ##################
    print(f"Using device: {configs.device}")

    # choose model architecture
    match configs.arch.lower():
        case "resnet18":
            model = models.resnet18()
        case "resnet34":
            model = models.resnet34()
        case "resnet50":
            model = models.resnet50()
        case _:
            raise ValueError(f"Model {configs.arch} not supported")

    # replace last layer with custom layer
    model.fc = nn.Linear(model.fc.in_features, configs.num_classes)

    # load checkpoint if provided
    if configs.checkpoint is not None:
        model.load_state_dict(torch.load(configs.checkpoint, map_location="cpu"))

    # print(model)
    model.to(configs.device)

    # initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)

    # initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=configs.epochs * len(train_dataloader), eta_min=1e-6 / configs.lr
    )

    #################
    ## TRAIN MODEL ##
    #################
    mlflow.set_tracking_uri("http://tularosa.sci.utah.edu:5000")
    mlflow.set_experiment("kiddo")
    mlflow.start_run(run_name=configs.name)
    mlflow.log_params(vars(configs))

    if not configs.skip_train:
        METRIC = "val_acc"
        best_metric = 0
        for epoch in range(configs.epochs):
            train_stats = train_loop(train_dataloader, model, optimizer, scheduler, configs)
            val_stats = val_loop(val_dataloader, model, configs)
            mlflow.log_metrics(train_stats | val_stats, step=epoch)

            print(
                f"epoch {epoch+1}/{configs.epochs} -- train acc: {train_stats['train_acc']*100:.2f}%, train loss: {train_stats['train_loss']:.4f}, val acc: {val_stats['val_acc']*100:.2f}%"
            )

            # save "best" model
            if val_stats[METRIC] > best_metric:
                best_metric = val_stats[METRIC]
                torch.save(model.state_dict(), os.path.join(configs.root, "best.pth"))

            # save last model
            torch.save(model.state_dict(), os.path.join(configs.root, "last.pth"))

        # load best model for final checks and saving
        model.load_state_dict(torch.load(os.path.join(configs.root, "best.pth")))

    # do one final validation loop on the best model or loaded model
    final_stats = val_loop(val_dataloader, model, configs)
    print(f"val acc: {final_stats['val_acc']*100:.2f}%")

    print("Done!")

    ################
    ## SAVE MODEL ##
    ################
    model.to("cpu")
    mlflow.pytorch.log_model(model, "model", conda_env="env.yaml")
    mlflow.log_artifacts(configs.root)
