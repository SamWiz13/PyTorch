import  torch
import  numpy as np

data =np.loadtxt('data.csv',delimiter=',',dtype=np.float32)

x_data = torch.from_numpy(data[:768, :8])
y_data = torch.from_numpy(data[:768, [8]])


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

for epoch in range(50000):
    y_predict = model(x_data)
    loss = criterion(y_predict, y_data)
    # if epoch % 1000 == 0:
        # print(f"Epoch-{epoch} | Loss: {loss.item()}")
    optim.zero_grad()
    loss.backward()
    optim.step()
print(loss.item())

test_data = np.loadtxt("data.csv",delimiter=',',dtype=np.float32)
test_x = torch.from_numpy(test_data[:2000,:8])
y_data = test_data[:2000,8]

S=0
for i in range(test_data.shape[0]):
    y_bash = model(test_x[i])
    result =1 if y_bash>0.5 else 0
    if result != y_data[i]:
        print(y_bash,test_x[i],i)
        S+=1
print((test_data.shape[0] - S)/test_data.shape[0])

