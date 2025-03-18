import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# Creating a baseline CNN on MNIST dataset.
#Checking Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device:{device}")

#Data Loading and Preprocessing - processes MNIST images into matrices for the model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081)) # values are the mean and std for the MNIST dataset

])

#Download and load the training and test sets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform = transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False, download= True, transform= transform )


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True )
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1000, shuffle= False)


# Defining the base CNN Model 

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # inputting the 1x28x28 images
        self.conv1 = nn.Conv2d(1,32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64x28x28
        self.pool   = nn.MaxPool2d(2, 2)  # Reduces spatial dims by factor of 2
        self.fc1    = nn.Linear(64 * 7 * 7, 128)  # after 2 pooling operations, image dims become 7x7
        self.fc2    = nn.Linear(128, 10)  # 10 classes for MNIST
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = BaselineCNN().to(device)
print(model)

# 3. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()  # reset gradients
        output = model(data)   # forward pass
        loss = criterion(output, target)
        loss.backward()        # backpropagation
        optimizer.step()       # update weights
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # print every 100 mini-batches
            print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] - Loss: {running_loss/100:.4f}")
            running_loss = 0.0

# 5. Evaluation Function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

# 6. Run Training and Evaluation
num_epochs = 5  # You can experiment with more epochs if time allows

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)

    