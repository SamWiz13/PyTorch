from torch import nn,optim,cuda,from_numpy
from torch.utils import data
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

b_size = 64
train_dataset = datasets.MNIST(root='/mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='/mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

train_data = data.DataLoader(train_dataset, batch_size=b_size, shuffle=True)
test_data = data.DataLoader(test_dataset, batch_size=b_size, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        return self.l5(x)


model = Model()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimize = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for b_i, data_ in enumerate(train_data):
        data_, label = data_[0].to(device), data_[1].to(device)
        optimize.zero_grad()
        bashorat = model(data_)
        xato = criterion(bashorat, label)
        xato.backward()
        optimize.step()

        if b_i % 10 == 0:
            print(f"Epoch - {epoch + 1} | Batch - {b_i} {b_i * len(data_)}/{len(train_data.dataset)} \
            | Loss: {xato.item()}")


def test():
    model.eval()
    Xato = 0;
    Togri = 0
    for rasm, label in test_data:
        rasm, label = rasm.to(device), label.to(device)
        bashorat = model(rasm)
        Xato += criterion(bashorat, label).item()
        natija = bashorat.data.max(1, keepdim=True)[1]
        Togri += natija.eq(label.data.view_as(natija)).cpu().sum()
    Xato /= (len(test_data.dataset))
    print(f"Aniqlik : {Togri / len(test_data.dataset)}, Foizda {100.0 * Togri / len(test_data.dataset)}%")


for epoch in range(10):
    train(epoch)
    test()