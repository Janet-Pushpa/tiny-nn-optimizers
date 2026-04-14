import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os



# Define Sinu-ReLU

class SinuReLU(nn.Module):

    def __init__(self):
        super(SinuReLU, self).__init__()
    def forward(self, x):
        return torch.maximum(torch.zeros_like(x), x) + torch.sin(x) * torch.exp(-torch.abs(x))

# Simple Neural Network
class Net(nn.Module):
    def __init__(self, activation_fn):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.act = activation_fn
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def train_and_evaluate(activation_name, activation_fn, train_loader, test_loader, epochs=3):
    print(f"\nTraining with {activation_name}...")

    model = Net(activation_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 400 == 0:
                print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
    duration = time.time() - start_time
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"{activation_name} Accuracy: {accuracy:.2f}%, Time: {duration:.2f}s")
    return accuracy, duration

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Use a subset of MNIST for speed
    data_path = './mnist_data'
    if not os.path.exists(data_path):

        os.makedirs(data_path)
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    # Limit to 10000 training samples for quick comparison

    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    results = {}
    # Test ReLU
    results['ReLU'] = train_and_evaluate('ReLU', nn.ReLU(), train_loader, test_loader)
    # Test Sinu-ReLU
    results['Sinu-ReLU'] = train_and_evaluate('Sinu-ReLU', SinuReLU(), train_loader, test_loader)
    print("\n--- Final Results ---")
    for name, (acc, dur) in results.items():
        print(f"{name}: Accuracy = {acc:.2f}%, Duration = {dur:.2f}s")