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