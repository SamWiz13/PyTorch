import  torch
import  numpy as np
from torch.utils.data import Dataset, DataLoader


class Datamiz(Dataset):
    def __init__(self):
        train_data = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = train_data.shape[0]
        self.x_data = torch.from_numpy(train_data[:, :-1])
        self.y_data = torch.from_numpy(train_data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = Datamiz()
train_dataset = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(8, 6)
        self.lin2 = torch.nn.Linear(6, 4)
        self.lin3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        res1 = self.relu(self.lin1(x))
        res2 = self.relu(self.lin2(res1))
        res3 = self.sigmoid(self.lin3(res2))
        return res3


model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optim = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(4):
    for x, data in enumerate(train_dataset, 0):
        x_dat, y_dat = data
        y_bash = model(x_dat)
        loss = criterion(y_bash, y_dat)
        print(f"E-{epoch} : | Loss : {loss.item()}")
        optim.zero_grad()
        loss.backward()
        optim.step()

test_data = np.loadtxt("data.csv",delimiter=',',dtype=np.float32)
test_x = torch.from_numpy(test_data[:2000,:8])
y_data = test_data[:2000,8]

S=0
for i in range(test_data.shape[0]):
    y_bash = model(test_x[i])
    result =1 if y_bash>0.5 else 0
    if result != y_data[i]:
        S+=1
print((test_data.shape[0] - S)/test_data.shape[0])

Y = np.array([1,0,0])
Y_bash = np.array([0.99,0.005,0.005])
print(f"Loss : | {np.sum(-Y * np.log(Y_bash))}")

loss = torch.nn.CrossEntropyLoss()
Y = torch.tensor([0],requires_grad=False)
Y_bash = torch.tensor([[10,0.001,0.001]])

fault = loss(Y_bash,Y)

print("Xatolik :",fault.item())

Y = torch.tensor([0,2,1],requires_grad=False)

Y_bash = torch.tensor([[10,0.001,0.001],
                       [0.01,0.01,10],
                       [0.1,10,0.1]])

xato = loss(Y_bash,Y)
print(xato.item())