from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nibabel as nib
from model import *
from sklearn.model_selection import train_test_split

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_shape=(128, 128, 128)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_shape = target_shape
        self.image_filenames = sorted(os.listdir(image_dir))  # Assuming filenames are consistent

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_filename = self.image_filenames[idx].replace('imagesTr', 'labelsTr')
        label_path = os.path.join(self.label_dir, label_filename)

        # Load image and label
        image = self.load_nifti_image(img_path)
        label = self.load_nifti_image(label_path)

        if image is None or label is None:
            return self.__getitem__((idx + 1) % len(self.image_filenames))



            # Resize or pad image and label to the target shape
        image = self.resize_or_pad(image, self.target_shape)
        label = self.resize_or_pad(label, self.target_shape)

        image = np.expand_dims(image, axis=0)

        if label.ndim == 0:
            label = np.expand_dims(label, axis=0)

        label = label.astype(np.int64)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)


        return image, label
    def load_nifti_image(self, path):
        try:
            nifti_img = nib.load(path)
            image_array = nifti_img.get_fdata(dtype=np.float32)
            return image_array
        except EOFError as e:
            print(f"Error loading file {path}: {e}")
            return None

    def resize_or_pad(self, image, target_shape):
        """Resize or pad an image to the target shape."""
        current_shape = image.shape

        # If the image is smaller than the target shape, pad it
        pad_size = [(0, max(0, target_dim - current_dim)) for current_dim, target_dim in zip(current_shape, target_shape)]
        padded_image = np.pad(image, pad_size, mode='constant')

        # If the image is larger than the target shape, center-crop it
        crop_start = [(current_dim - target_dim) // 2 if current_dim > target_dim else 0 for current_dim, target_dim in zip(current_shape, target_shape)]
        cropped_image = padded_image[crop_start[0]:crop_start[0] + target_shape[0],
                                     crop_start[1]:crop_start[1] + target_shape[1],
                                     crop_start[2]:crop_start[2] + target_shape[2]]

        return cropped_image


transform = None

# data_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip()
# ])

# Create dataset and dataloaders
image_dir = '/home/ubuntu/liver/data/Task03_Liver_rs/imagesTr'
label_dir = '/home/ubuntu/liver/data/Task03_Liver_rs/labelsTr'
target_shape = (128, 128, 128)  # Choose a common shape for all images

dataset = MedicalImageDataset(image_dir=image_dir, label_dir=label_dir, transform=transform, target_shape=target_shape)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

# Test the DataLoader
for i, (img, mask) in enumerate(val_loader):
    print("val Image shape:", img.shape)
    print("val Label shape:", mask.shape)
    if i == 2:  # Limit output for testing
        break



class Segmenter(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        mask = mask.long()  # Ensure the labels are of type long
        y_hat = self(img)
        loss = self.loss_fn(y_hat, mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        mask = mask.long()  # Ensure the labels are of type long
        y_hat = self(img)
        loss = self.loss_fn(y_hat, mask)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return [self.optimizer]



model = Segmenter()

checkpoint_callback = ModelCheckpoint(

    monitor='val_loss',
    save_top_k=2,
    mode = 'min'
)



trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    logger=TensorBoardLogger('./logs'),
    callbacks=checkpoint_callback,
    max_epochs = 10
)


trainer.fit(model, train_loader, val_loader)

