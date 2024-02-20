from torchvision import models, transforms
import torch.nn as nn
import logging
from models import ResNet18

def get_model(config, num_classes):
    model_name = config["model_params"]["model_name"].lower()
    if model_name == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Model name should be in [ResNet18, ResNet50], but got {model_name} instead.")

    return model

def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train, transform_test


class Logger:
    def __init__(self, log_file='training.log'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file, mode='a')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def print_and_log(self, message):
        print(message)
        self.logger.info(message)
