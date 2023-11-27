import os
import numpy as np
import torch.utils.data
from sklearn.svm import SVC
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from sklearn.decomposition import PCA
import cv2


class FreshRottenDataset(Dataset):
    def __init__(self, image_folder, transform=None, ):
        self.image_paths = []
        self.transform = transform
        classes = []
        for dir1 in os.listdir(image_folder):
            classes.append(dir1)
            for file in os.listdir(os.path.join(image_folder, dir1)):
                self.image_paths.append(os.path.normpath(os.path.join(image_folder, dir1, file)))
        self.class_to_idx = {j: i for i, j in enumerate(classes)}
        self.idx_to_class = {i: j for i, j in enumerate(classes)}
        self.num_classes = len(classes)
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
        label = image_filepath.split(os.path.sep)[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)

        return image, label


transforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

train_dataset = FreshRottenDataset(image_folder=os.path.join('.', 'apple_train'), transform=transforms)
test_dataset = FreshRottenDataset(image_folder=os.path.join('.', 'apple_train'), transform=transforms)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_red_by_class = [[] for _ in range(train_dataset.num_classes)]
train_blue_by_class = [[] for _ in range(train_dataset.num_classes)]
train_green_by_class = [[] for _ in range(train_dataset.num_classes)]
for image, target in train_dataset:
    img = image.permute(1, 2, 0).numpy()
    b, g, r = cv2.split(img)
    train_red_by_class[target].append(r.flatten())
    train_blue_by_class[target].append(b.flatten())
    train_green_by_class[target].append(g.flatten())
train_num_by_class = [len(images) for images in train_red_by_class]
train_red, train_blue, train_green = np.concatenate(train_red_by_class), np.concatenate(
    train_blue_by_class), np.concatenate(train_green_by_class)

test_red_by_class = [[] for _ in range(test_dataset.num_classes)]
test_blue_by_class = [[] for _ in range(test_dataset.num_classes)]
test_green_by_class = [[] for _ in range(test_dataset.num_classes)]
for image, target in test_dataset:
    img = image.permute(1, 2, 0).numpy()
    b, g, r = cv2.split(img)
    test_red_by_class[target].append(r.flatten())
    test_blue_by_class[target].append(b.flatten())
    test_green_by_class[target].append(g.flatten())
test_num_by_class = [len(images) for images in test_red_by_class]
test_red, test_blue, test_green = np.concatenate(test_red_by_class), np.concatenate(
    test_blue_by_class), np.concatenate(test_green_by_class)

pca = PCA(n_components=1)
trans_train_red = pca.fit_transform(train_red)
trans_test_red = pca.transform(test_red)
trans_train_blue = pca.fit_transform(train_blue)
trans_test_blue = pca.transform(test_blue)
trans_train_green = pca.fit_transform(train_green)
trans_test_green = pca.transform(test_green)

# Prep the train data for svc
X_train = np.hstack((trans_train_red, trans_train_blue, trans_train_green))
Y_train = [0] * train_num_by_class[0] + [1] * train_num_by_class[1]
# Prep the test data for svc
X_test = np.hstack((trans_test_red, trans_test_blue, trans_test_green))
Y_test = [0] * test_num_by_class[0] + [1] * test_num_by_class[1]


svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)
xy = np.linspace(-50, 50, 100)
x_val, y_val = np.meshgrid(xy, xy)
# Equation of plane to solve for z
z = lambda x, y: (svc.intercept_[0] + svc.coef_[0][0] * x + svc.coef_[0][1] * y) / -svc.coef_[0][2]
print(f"Decision Boundary Equation: {svc.intercept_[0]}+{svc.coef_[0][0]}x+{svc.coef_[0][1]}y+{svc.coef_[0][2]}z=0")

# Plot Train
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax = fig.add_subplot()
# Plot decision boundary plane
ax.plot_surface(x_val, y_val, z(x_val, y_val), color='y')
# Plot test data
ax.scatter(trans_test_red[:test_num_by_class[0]], trans_test_blue[:test_num_by_class[0]],
           trans_test_green[:test_num_by_class[0]],
           facecolors='none', edgecolors='r', label=test_dataset.idx_to_class[0])
ax.scatter(trans_test_red[test_num_by_class[0]:], trans_test_blue[test_num_by_class[0]:],
           trans_test_green[test_num_by_class[0]:],
           facecolors='none', edgecolors='b', label=test_dataset.idx_to_class[1])
# Plot training data
# ax.scatter(trans_train_red[:train_num_by_class[0]], trans_train_blue[:train_num_by_class[0]], trans_train_green[:train_num_by_class[0]],
#            facecolors='none', edgecolors='r', label=train_dataset.idx_to_class[0])
# ax.scatter(trans_train_red[train_num_by_class[0]:], trans_train_blue[train_num_by_class[0]:], trans_train_green[train_num_by_class[0]:],
#            facecolors='none', edgecolors='b', label=train_dataset.idx_to_class[1])

ax.set_xlabel('Red')
ax.set_ylabel('Blue')
ax.set_zlabel('Green')
plt.legend()
plt.title('Banana')
plt.show()

# Compute test accuracy using decision boundary classification
test_score = svc.score(X_test, Y_test)

error = 0
for i in range(test_dataset.__len__()):
    _, label = test_dataset[i]
    r_val, b_val, g_val = trans_test_red[i], trans_test_blue[i], trans_test_green[i]
    prediction = (svc.intercept_[0] + svc.coef_[0][0] * r_val + svc.coef_[0][1] * b_val + svc.coef_[0][2] * g_val) < 0
    if label != prediction:
        error += 1
print(f"Test Accuracy: {error/test_dataset.__len__()} SVC Score: {test_score} Errors: {error}")

# plt.figure(figsize=(10,10))
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(dataset.__getitem__(i+100)[0].reshape(3, 200, 200).permute(1, 2, 0))
# plt.show()
