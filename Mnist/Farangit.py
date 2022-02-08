import  matplotlib.pyplot as plt
import  numpy as np

celsius    = np.array([-15.0,-10.0,-5.0, 0.0,  5.0, 10.0])
fahrenheit = np.array([5.0, 14.0, 23.0, 32.0, 41.0,50.0])

w =1
b =1
def forward(x):
    return x * w + b

# Xatolikni topish funksiyasi
def loss(x,y):
    y_predict = forward(x)
    return (y -y_predict) ** 2

lr = 0.001  # O'rganish qadamini belgilash

# Training loop
for epoch in range(10000):
    for x,y in zip(celsius,fahrenheit):
        y_predict = forward(x)
        fault = loss(x,y)
        w -=- 2 * lr * x * (y - y_predict)
        b -= - 2 * lr * (y - y_predict)
    print(f"E :{epoch} | F : {fault} | W : {w}")
print("Model = ", w)
print("Biar :",b)