"""
################################################################################
#                     Subtract noise from a sine wave                          #
################################################################################
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
import numpy as np


# ---------- build a noisy sine wave ---------- #
sine_wave = torch.sin(torch.arange(0, 100, 0.1))
witness = torch.sin(4.0 * torch.arange(0, 100, 0.1))
observed = sine_wave + 0.2 * torch.sin(4.0 * torch.arange(0, 100, 0.1) + np.pi)
INPUTS = 1  # number of inputs into network

# ------------ Plot the input data ----------- #
plt.plot(sine_wave.numpy()[:100], label='target')
plt.plot(witness.numpy()[:100], label='witness')
plt.plot(observed.numpy()[:100], label='observed')
plt.legend()
plt.savefig('dirty_sine.png')
plt.close()


# --------- Build the Network --------------- #
class SineNetwork(torch.nn.Module):
    def __init__(self):
        super(SineNetwork, self).__init__()
        self.linear = torch.nn.Linear(INPUTS, 1)  # couldnt be easier!

    def forward(self, x):
        return self.linear(x)

sine_network = SineNetwork()  # create instance


# -------- Get loss and optimizer -------- #
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(sine_network.parameters(), lr=0.001)
test_size = np.int(witness.size(0) / 2.0)
epochs = 20

loss_progress = []
for epoch in range(epochs):
    temp_loss = 0
    for i in range(test_size - INPUTS + 1):
        out = sine_network(witness[i:i+INPUTS])
        loss = criterion(out, observed[i])
        sine_network.zero_grad()
        loss.backward()
        optimizer.step()

        temp_loss += loss.item()
    loss_progress.append(temp_loss / (test_size - INPUTS + 1))

# ------------ Testing ----------------- #
predict = np.zeros(test_size - INPUTS)
for i in range(test_size - INPUTS):
    predict[i] = sine_network(witness[test_size + i: test_size + i + INPUTS]).detach().numpy()

# ------------ Make plots -------------- #
cleaned_prediction = observed[test_size:-INPUTS].numpy() - predict
plt.plot(cleaned_prediction, label='cleaned prediction')
plt.plot(sine_wave.numpy()[test_size:-INPUTS], label='target')
plt.legend()
plt.savefig('cleaned.png')
plt.close()

plt.plot((cleaned_prediction - sine_wave.numpy()[test_size:-INPUTS])[:100], label='residual')
plt.legend()
plt.savefig('residual.png')
plt.close()

plt.plot(loss_progress)
plt.xlabel('epoch')
plt.title('Loss')
plt.savefig('sine_loss.png')
plt.close()
