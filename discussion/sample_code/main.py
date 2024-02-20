import argparse
from helpers import get_model, get_transforms, Logger
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

import os


def train(model, train_loader, criterion, optimizer, device, epoch, logger):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    logger.info(
        f"At Epoch {epoch}: Loss: {running_loss / len(train_loader):.3f}, Accuracy: {100 * correct / total:.2f}%"
    )


def test(model, test_loader, criterion, device, epoch, logger, best_acc):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logger.info(
        f"At Epoch {epoch}: Accuracy of the network on the test images: {accuracy:.2f}%"
    )
    if accuracy > best_acc:
        best_acc = accuracy
        logger.info(f"Best Test Accuracy so far: {best_acc:.2f}%")
        logger.info("Saving the model")
        torch.save(model.state_dict(), os.path.join(config["run_path"], 'best_model.pth'))

    return best_acc


def main(config, logger):
    message = "The configuration is: " + str(config)
    logger.info(message)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config, num_classes=config["model_params"]["num_classes"])
    model.to(device)

    transform_train, transform_test = get_transforms()
    train_dataset = datasets.CIFAR10(
        root=config["data_dir"], train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config["trainig_params"]["batch_size"],
        shuffle=True,
    )
    test_dataset = datasets.CIFAR10(
        root=config["data_dir"], train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config["trainig_params"]["batch_size"],
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["trainig_params"]["learning_rate"],
        weight_decay=config["trainig_params"]["weight_decay"],
    )

    best_acc = 0
    for epoch in range(config["trainig_params"]["num_epochs"]):
        train(model, train_loader, criterion, optimizer, device, epoch, logger)
        best_acc = test(model, test_loader, criterion, device, epoch, logger, best_acc)

    logger.info(f"Finished Training with best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":

    ## Get the arguments from the command line
    parser = argparse.ArgumentParser(description="Arguments for sample code")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--wd", type=float, help="Weight decay")
    args = parser.parse_args()

    # Overwrite the config file with the arguments
    # use the config file to pass in parameters in the main function
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

    config["trainig_params"]["batch_size"] = (
        args.bs if args.bs else config["trainig_params"]["batch_size"]
    )
    config["trainig_params"]["learning_rate"] = (
        args.lr if args.lr else config["trainig_params"]["learning_rate"]
    )
    config["trainig_params"]["num_epochs"] = (
        args.epochs if args.epochs else config["trainig_params"]["num_epochs"]
    )
    config["model_params"]["model_name"] = (
        args.model_name if args.model_name else config["model_params"]["model_name"]
    )
    config["trainig_params"]["weight_decay"] = (
        args.wd if args.wd else config["trainig_params"]["weight_decay"]
    )

    run_name = f"{config['model_params']['model_name']}_bs_{config['trainig_params']['batch_size']}_lr_{config['trainig_params']['learning_rate']}_wd_{config['trainig_params']['weight_decay']}_epochs_{config['trainig_params']['num_epochs']}"
    # create a folder for the run
    config["run_path"] = f"{config['run_dir']}/{run_name}"
    if not os.path.exists(config["run_path"]):
        os.makedirs(config["run_path"])
    config["log_file"] = f"{config['run_path']}/training.log"

    logger = Logger(config["log_file"])

    main(config, logger)
