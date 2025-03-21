import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro.optim as optim

# -----------------------
# 1. Environment Setup and Data Loading
# -----------------------
torch.manual_seed(0)
pyro.set_rng_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transformations: convert images to tensors and normalize them.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load MNIST training and test datasets.
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -----------------------
# 2. Define the Bayesian CNN Model using PyroModule
# -----------------------
class BayesianCNN(PyroModule):
    def __init__(self):
        super().__init__()
        # Bayesian Convolutional Layer 1: from 1 channel to 32 channels.
        self.conv1 = PyroModule[nn.Conv2d](1, 32, kernel_size=3, padding=1)
        self.conv1.weight = PyroSample(
            dist.Normal(0., 1.).expand(self.conv1.weight.shape).to_event(self.conv1.weight.dim())
        )
        self.conv1.bias = PyroSample(
            dist.Normal(0., 1.).expand(self.conv1.bias.shape).to_event(1)
        )
        
        # Bayesian Convolutional Layer 2: from 32 channels to 64 channels.
        self.conv2 = PyroModule[nn.Conv2d](32, 64, kernel_size=3, padding=1)
        self.conv2.weight = PyroSample(
            dist.Normal(0., 1.).expand(self.conv2.weight.shape).to_event(self.conv2.weight.dim())
        )
        self.conv2.bias = PyroSample(
            dist.Normal(0., 1.).expand(self.conv2.bias.shape).to_event(1)
        )
        
        # Deterministic Max Pooling Layer: reduces spatial dimensions.
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bayesian Fully Connected Layer 1: from flattened features to 128 neurons.
        self.fc1 = PyroModule[nn.Linear](64 * 7 * 7, 128)
        self.fc1.weight = PyroSample(
            dist.Normal(0., 1.).expand(self.fc1.weight.shape).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0., 1.).expand(self.fc1.bias.shape).to_event(1)
        )
        
        # Bayesian Fully Connected Layer 2: from 128 neurons to 10 output classes.
        self.fc2 = PyroModule[nn.Linear](128, 10)
        self.fc2.weight = PyroSample(
            dist.Normal(0., 1.).expand(self.fc2.weight.shape).to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0., 1.).expand(self.fc2.bias.shape).to_event(1)
        )
        
        # Deterministic ReLU activation.
        self.relu = nn.ReLU()
    
    def forward(self, x, y=None):
        # Pass input through first conv layer, activation, and pooling.
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        # Pass through second conv layer, activation, and pooling.
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the output for the fully connected layers.
        x = x.view(-1, 64 * 7 * 7)
        # First fully connected layer with activation.
        x = self.relu(self.fc1(x))
        # Final fully connected layer produces logits.
        logits = self.fc2(x)
        
        # Probabilistic observation:
        # 'pyro.plate' handles batching; we assume targets come from a Categorical distribution.
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

# Instantiate and display the model.
model = BayesianCNN().to(device)
print(model)

# -----------------------
# 3. Setup SVI for Training the Bayesian CNN
# -----------------------
# Create an automatic guide (variational distribution) that approximates the posterior.
guide = AutoDiagonalNormal(model)

# Setup the optimizer and SVI using Trace_ELBO as the loss.
pyro_optimizer = optim.Adam({"lr": 0.001})
svi = SVI(model, guide, pyro_optimizer, loss=Trace_ELBO())

# -----------------------
# 4. Training Loop
# -----------------------
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # svi.step runs the model and guide, returning the ELBO loss for the batch.
        loss = svi.step(data, y=target)
        epoch_loss += loss
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)} Loss: {loss/len(data):.4f}")
    avg_loss = epoch_loss / len(train_dataset)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

# -----------------------
# 5. Evaluation using Predictive Posterior
# -----------------------
# Setup Predictive to sample from the posterior predictive distribution.
predictive = Predictive(model, guide=guide, num_samples=10)
model.eval()
correct = 0
total = 0

for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    # Generate posterior predictive samples.
    samples = predictive(data)
    # 'obs' in the samples were drawn from a Categorical, so for simplicity,
    # we take the first sample as the predicted label.
    preds = samples["obs"][0]
    correct += (preds == target).sum().item()
    total += target.size(0)

accuracy = 100. * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
