import  torch
import  numpy as np

x_data =torch.tensor([[1.0,5.0],
                      [2.0,3.0],
                      [3.0,1.0],
                      [4.0,2.0]])
y_data =torch.tensor([[1.0],
                      [1.0],
                      [0.0],
                      [1.0]])

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2,1)
    def forward(self,x):
        y_predict =self.linear(x)
        return y_predict

model =Model()

criterion =torch.nn.MSELoss(reduction='sum')
optimize =torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_predict =model(x_data)
    fault =criterion(y_predict,y_data)
    print(f'E {epoch} | f {fault.item}')
    optimize.zero_grad()
    optimize.step()

print(model.forward(torch.tensor([2.0,3.0])))
