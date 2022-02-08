import  torch
import  numpy as np
import  matplotlib.pyplot as plt

x_data =torch.tensor([[-15.0],
                      [-10.0],
                      [-5.0],
                      [0.0],
                      [5.0],
                      [10.0]])
y_data =torch.tensor([[5.0],
                      [14.0],
                      [23.0],
                      [32.0],
                      [41.0],
                      [50.0]])

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        y_predict =self.linear(x)
        return y_predict

model =Model()

criterion =torch.nn.MSELoss(reduction='sum')
optimize =torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(10000):
    y_predict =model(x_data)
    fault =criterion(y_predict,y_data)
    print(f'E {epoch} | f {fault.item()}')
    optimize.zero_grad()
    optimize.step()

print(model.forward(torch.tensor([0.0])))
