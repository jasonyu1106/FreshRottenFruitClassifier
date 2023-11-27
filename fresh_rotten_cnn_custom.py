import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

print(torch.cuda.is_available())

# Basic settings
root = 'data'
num_classes = 2
batch_size = 16  # size of the mini-batch we will use when calculating gradients
num_epochs = 25  # number of times we iterate through the full dataset
device = 'cuda:0'  # if you have a GPU available you might use 'cuda:0' otherwise 'cpu'
print_interval = 50  # how often to print the loss during training


class FreshRottenDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = []
        self.transform = transform
        self.batch_size = batch_size
        for dir1 in os.listdir(image_folder):
            for file in os.listdir(os.path.join(image_folder, dir1)):
                self.image_paths.append(os.path.normpath(os.path.join(image_folder, dir1, file)))
        sizes = []
        for file_path in self.image_paths:
            img = Image.open(file_path)
            sizes.append(img.size)
        print(min(sizes))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath).convert('RGB')
        label = int(image_filepath.split(os.path.sep)[-2].startswith('fresh'))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


transforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

train_dataset = FreshRottenDataset(image_folder=os.path.join('.', 'dataset/train'), transform=transforms)
test_dataset = FreshRottenDataset(image_folder=os.path.join('.', 'dataset/test'), transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Make model
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    # nn.Sigmoid(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    # nn.Sigmoid(),
    nn.MaxPool2d(2),

    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # nn.Sigmoid(),
    nn.MaxPool2d(2),

    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # nn.Sigmoid(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(16384, 4096),
    nn.ReLU(inplace=True),
    # nn.Sigmoid(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    # nn.Sigmoid(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 1)
)
model.to(device)

# Define an optimizer
optimizer = torch.optim.Adam((model.parameters()))

loss_func = nn.BCEWithLogitsLoss()

def run_test_evaluation(eval_model):
    eval_model.eval()
    with torch.no_grad():
        test_correct = 0
        for test_data, test_targets in test_loader:
            # Make sure data is on the same device as the model
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)
            test_targets = test_targets.unsqueeze(1)
            test_targets = test_targets.float()

            # Run forward pass
            eval_logits = eval_model(test_data)
            # Compute loss
            eval_loss = loss_func(eval_logits, test_targets)

            # Get class predictions
            test_probs = torch.sigmoid(eval_logits)
            test_predicted_labels = (test_probs > 0.5).int()
            test_correct += (test_predicted_labels == test_targets).int().sum()

    # return final test accuracy and test loss
    return 100 * test_correct.item() / len(test_dataset), eval_loss.item()


step = 0  # track how many training iterations we've been through
train_losses = []  # will be used to store loss values after each epoch
train_accuracies = []
test_accuracies = []
test_losses = []
for epoch in range(num_epochs):
    correct = 0
    model.train()  # sets model to training mode
    for data, targets in train_loader:
        # Move data and targets to the same device as the model
        data = data.to(device)
        targets = targets.to(device)
        targets = targets.unsqueeze(1)
        targets = targets.float()

        # Zero-out gradients - torch accumulates gradients by dfault,
        # so we have to manually reset them in each iteration
        optimizer.zero_grad()

        # Run forward pass - model returns the logits with shape (batch_size, num_classes)
        logits = model(data)

        # Compute loss
        loss = loss_func(logits, targets)

        # Perform backpropagation and update model parameters
        loss.backward()
        optimizer.step()

        # (Optional) store and print loss
        if not (step + 1) % print_interval:
            print('[epoch: {}, step: {}, loss: {}]'.format(epoch, step, loss.item()))

        # Update test accuracy
        probs = torch.sigmoid(logits)
        predicted_labels = (probs > 0.5).int()
        correct += (predicted_labels == targets).int().sum()
        step += 1

    train_accuracies.append(100 * correct.item() / len(train_dataset))
    print(train_accuracies[-1])
    train_losses.append(loss.item())

    # Calculate test accuracy and test loss while training
    test_accuracy, test_loss = run_test_evaluation(model)
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)
    print(test_accuracies[-1])


state_dict = model.state_dict()
torch.save(state_dict, 'fresh_rotten_model_weights.pth')
# state_dict_loaded = torch.load('fresh_rotten_model_weights.pth')
# model.load_state_dict(state_dict_loaded)
# model.eval()

plt.figure()
plt.plot(range(len(train_accuracies)), train_accuracies, '-o')
plt.xticks(range(len(train_accuracies)))
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy (%)')
plt.savefig('NN (RMSProp) Training Accuracy Plot')

plt.figure()
plt.plot(range(len(train_losses)), train_losses, '-o')
plt.xticks(range(len(train_losses)))
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.savefig('NN Training Loss Plot')

plt.figure()
plt.plot(range(len(test_accuracies)), test_accuracies, '-o')
plt.xticks(range(len(test_accuracies)))
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.savefig('NN (RMSProp) Test Accuracy Plot')

plt.figure()
plt.plot(range(len(test_losses)), test_losses, '-o')
plt.xticks(range(len(test_losses)))
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.savefig('NN Test Loss Plot')

final_test_accuracy, final_test_loss = run_test_evaluation(model)
print(f"Test accuracy on trained model: {final_test_accuracy}")
print(f"Test loss on trained model: {final_test_loss}")


# Application on timelapse images
# timelapse_dataset = FreshRottenDataset(image_folder=os.path.join('.', 'timelapse_data_banana'), transform=transforms)
# timelapse_loader = DataLoader(timelapse_dataset, batch_size=1, shuffle=False)
# results = []
# with torch.no_grad():
#     test_correct = 0
#     for test_data, test_targets in timelapse_loader:
#         test_data = test_data.to(device)
#         test_logits = model(test_data)
#         probs = torch.sigmoid(test_logits)
#         print(probs)
#         label = (probs > 0.5).int()
#         results.append(label.item())
#
# print(results)
