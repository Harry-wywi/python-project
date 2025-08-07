import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  #拿数据
# 定义网络参数并进行初始化
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# nn.Parameter 申明这是torch的参数, 可以不加
W1 = nn.Parameter(
        torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.Parameter(
       torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
# 激活函数
def relu(X):
 a = torch.zeros_like(X)
 return torch.max(X, a)
# 定义模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')
# 开始训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

def evaluate_accuracy(net, data_iter):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            pred = y_hat.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / total

if __name__ == "__main__":
    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, test acc: {acc:.4f}')

    # 预测功能已移除或请自行实现
    # d2l.predict_ch3(net, test_iter)