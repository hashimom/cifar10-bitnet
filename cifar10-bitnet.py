import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor
from bitnet import replace_linears_in_pytorch_model

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(8 * 8 * 64, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2(x)

        x = x.view(-1, 8 * 8 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# Training
def train(net, optimizer, criterion, train_loader, device="cuda"):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    metrics = {"loss": 0., "acc": 0.}
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.long()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        metrics["loss"] = train_loss/(batch_idx+1)
        metrics["acc"] = 100.*correct/total

    return metrics


def test(net, criterion, test_loader, device="cuda"):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    metrics = {"loss": 0., "acc": 0.}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            metrics["loss"] = test_loss / (batch_idx + 1)
            metrics["acc"] = 100. * correct / total

    return metrics


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epoch', default=50, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--device', default="cuda", type=str, help='device (cuda or cpu)')
    args = parser.parse_args()

    net = Net()
    replace_linears_in_pytorch_model(net)
    net = net.to(args.device)

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)

    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + args.epoch):
        train_metrics = train(net, optimizer, criterion, train_loader, args.device)

        test_metrics = test(net, criterion, test_loader, args.device)

        print("[Epoch-%03d] train: ACC=%.2f / LOSS=%f  valid: ACC=%.2f / LOSS=%f" %
              (epoch, train_metrics["acc"], train_metrics["loss"], test_metrics["acc"], test_metrics["loss"]))


if __name__ == "__main__":
    main()
