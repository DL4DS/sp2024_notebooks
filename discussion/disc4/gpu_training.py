import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from prettytable import PrettyTable

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def name(self):
        return 'MLP'


class CNN(nn.Module):
    '''
    Convolutional Neural Network.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def name(self):
        return 'CNN'

def train_model(model, trainloader, loss_function, optimizer, device, model_name):
    total_time_start = time.time()


    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum
        # Set model to training mode
        model.train()

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done for {model_name} on {device}!")

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    avg_time_per_epoch = total_time / 5

    return total_time, avg_time_per_epoch

if __name__ == '__main__':
    torch.manual_seed(42)
    dataset = CIFAR10('/projectnb/ds598/materials/datasets/CIFAR', download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

    results = []
    devices = []
    if torch.cuda.is_available():
        devices.append('cuda')  # Add 'cuda' to indicate GPU availability
    devices.append('cpu')  # Add 'cpu' to indicate CPU availability

    for device in devices:
        for Model in [MLP, CNN]:
            model = Model()
            if device == 'cuda' and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model).to(device)  # Wrap model for multi-GPU support, move to GPU
            else:
                model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            model_name = model.module.name() if isinstance(model, nn.DataParallel) else model.name()  # Adjusted for DataParallel wrapped models
            total_time, avg_time_per_epoch = train_model(model, trainloader, nn.CrossEntropyLoss(), optimizer, device, model_name)
            device_label = "Multi-GPU" if isinstance(model, nn.DataParallel) else device.upper()
            results.append({
                "Model": model_name,
                "Device": device_label,
                "Total Time (seconds)": total_time,
                "Avg Time per Epoch": avg_time_per_epoch
            })

    # Sort results by Total Time (ascending)
    sorted_results = sorted(results, key=lambda x: x["Total Time (seconds)"])

    # Create PrettyTable
    table = PrettyTable()
    table.field_names = ["Model", "Device", "Total Time (seconds)", "Avg Time per Epoch"]
    for result in sorted_results:
        table.add_row([result["Model"], result["Device"], f"{result['Total Time (seconds)']:.2f}", f"{result['Avg Time per Epoch']:.2f}"])

    print(table)
    
    # Save the pretty table to a text file
    with open("model_performance_summary.txt", "w") as file:
        file.write(str(table))
