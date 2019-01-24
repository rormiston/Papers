import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np


# define the network
class LinearRegression(torch.nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # 2 in and 1 out

    def forward(self, x):
        return self.linear(x)


# create an instance of the network
net = LinearRegression()

# generate training / testing data
x_train = torch.rand(200, 2) * 5
y_train = torch.sum(x_train, dim=1)

x_test = torch.rand(20, 2) * 5
y_test = torch.sum(x_test, dim=1)

# set the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# train the network && collect the loss data for plotting
avg_loss = []
dod = []
for epoch in range(100):
    temp_sum = 0  # get loss for each input on each epoch
    temp_dod = []
    for j in range(x_train.shape[0]):
        out = net(x_train[j, :])
        loss = criterion(out, y_train[j])
        net.zero_grad()
        loss.backward()
        optimizer.step()

        # collect losses
        temp_sum += loss.item()
        temp_dod.append(loss.item())

    dod.append(temp_dod)
    avg_loss.append(temp_sum / 200)  # append the average loss

l0 = [x[0] for x in dod]
l1 = [x[1] for x in dod]
l2 = [x[2] for x in dod]

# show predictions to stdout
for ix in range(x_test.shape[0]):
    prediction = net(x_test[ix, :])
    print('[+] Predicted: {0:.3f}  Target: {1:.3f}  Error: {2:.3f}%'.format(prediction.item(), y_test[ix], np.abs(prediction.item() - y_test[ix]) * 100))

# plot the loss data
plt.style.use('dark_background')
plt.plot(list(range(100)), avg_loss, label='Average Loss')
plt.plot(list(range(100)), l0, label='Input 0')
plt.plot(list(range(100)), l1, label='Input 1')
plt.plot(list(range(100)), l2, label='Input 2')
plt.title('Average Loss for Linear Addition Network')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(alpha=0.6)
plt.xlim([0, 20])
plt.savefig('loss.png')

