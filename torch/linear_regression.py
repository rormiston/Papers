import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
import numpy as np


# create noisy linear training / testing data
# unsqueeze reshapes to (-1, 1) i.e., a column vector
observed = torch.arange(0, 5, 0.1).unsqueeze(1)
target = 3 * torch.arange(0, 5, 0.1).unsqueeze(1)
test_data = torch.arange(6, 10, 0.1).unsqueeze(1)

# build model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

net = LinearRegression()

# set loss / optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.001)

# train the model
loss_history = []
for epoch in range(5000):
    output = net(observed)
    loss = criterion(output, target)
    net.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

# Generate a prediction
prediction = net(test_data).detach().numpy()

# Get what we learned
slope = list(net.parameters())[0].item()
bias = list(net.parameters())[-1].item()

# plot it
test_target = 3 * test_data.numpy()
plt.plot(test_target, label='target $y(x) = 3x + b$')
plt.plot(prediction, label='prediction: $pred(x) = {0:.3f}x + {1:.3f}$'.format(slope, bias))
plt.title('Target vs Prediction')
plt.legend()
plt.savefig('linear_predition.png')
plt.close()

plt.plot(loss_history)
plt.title('Linear Regression Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.savefig('linear_loss.png')
plt.close()

plt.plot(test_target - prediction)
plt.title('Residual Linear Fit')
plt.savefig('linear_residual.png')
plt.close()


