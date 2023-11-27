import math
import os
import numpy as np
from keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.python.keras.optimizer_v2.adam import Adam
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from torchvision import transforms as T
from tensorflow.python.keras.metrics import BinaryAccuracy
from tensorflow.python.keras.losses import BinaryCrossentropy


class FreshRottenDataset(Sequence):
    def __init__(self, image_folder, batch_size, transform=None):
        self.image_paths = []
        self.transform = transform
        self.batch_size = batch_size
        classes = []
        for dir1 in os.listdir(image_folder):
            classes.append(dir1)
            for file in os.listdir(os.path.join(image_folder, dir1)):
                self.image_paths.append(os.path.normpath(os.path.join(image_folder, dir1, file)))
        self.num_classes = 2
        sizes = []
        for file_path in self.image_paths:
            img = Image.open(file_path)
            sizes.append(img.size)
        print(min(sizes))

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, idx):
        image_filepaths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = [Image.open(path).convert('RGB') for path in image_filepaths]
        labels = [int(path.split(os.path.sep)[-2].startswith('fresh')) for path in image_filepaths]
        if self.transform is not None:
            images = [np.array(self.transform(img).permute(1, 2, 0)) for img in images]
        return np.array(images), np.array(labels)


transforms = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

train_dataset = FreshRottenDataset(image_folder=os.path.join('.', 'dataset/train'), transform=transforms, batch_size=64)
test_dataset = FreshRottenDataset(image_folder=os.path.join('.', 'dataset/test'), transform=transforms, batch_size=64)
num_classes = 2

model = Sequential()
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3), classes=num_classes)
# base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3), classes=num_classes)
# base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3), classes=num_classes)
model.add(base_model)
model.layers[0].trainable = False
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=BinaryAccuracy())
model.fit(train_dataset, epochs=10)
model.summary()
model.save_weights("trained_frozen_base_resnet")

result = model.evaluate(test_dataset)
print(result)

# model.load_weights("trained_frozen_base_resnet")
# model.layers[0].trainable = True
# opt = Adam(learning_rate=0.0001)
# model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=BinaryAccuracy())
# model.fit(train_dataset, epochs=10)
# model.summary()
# result = model.evaluate(test_dataset)
# print(result)
