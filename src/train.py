from pathlib import Path
import torchio as tio
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import nibabel as nib
from model import *
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import os





# def load_nifti_image(file_path):
#     # Load the image
#     img = nib.load(file_path)
#     # Convert to numpy array
#     img_data = img.get_fdata()
#     # Convert to torch tensor
#     img_tensor = torch.tensor(img_data, dtype=torch.float32)
#     return img_tensor


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))  # Assuming filenames are consistent

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_filename = self.image_filenames[idx].replace('imagesTr', 'labelsTr')
        label_path = os.path.join(self.label_dir, label_filename)

        # Load image and label (assuming .nii or .nii.gz files)
        image = self.load_nifti_image(img_path)
        label = self.load_nifti_image(label_path)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)  # Apply the same transform to the label if necessary

        return image, label

    def load_nifti_image(self, path):
        """Load a NIfTI image and return a NumPy array."""
        nifti_img = nib.load(path)
        image_array = nifti_img.get_fdata(dtype=np.float32)  # Get data as NumPy array
        return image_array


# Example usage of the dataset

# Define a transform (assuming 2D images; modify for 3D)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Create dataset
# dataset = MedicalImageDataset(image_dir='/path/to/images', label_dir='/path/to/labels', transform=transform)
#
# # Split dataset into training and validation sets
# train_size = int(0.8 * len(dataset))  # 80% for training
# val_size = len(dataset) - train_size  # 20% for validation
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# # Create DataLoaders for training and validation datasets
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
#
# # Check shapes of images and labels in dataloaders
# for image, label in train_loader:
#     print("Train Image shape:", image.shape)
#     print("Train Label shape:", label.shape)
#     break
#
# for image, label in val_loader:
#     print("Val Image shape:", image.shape)
#     print("Val Label shape:", label.shape)
#     break







# Define dataset and data loaders
image_dir = '/home/ubuntu/liver/data/Task03_Liver_rs/imagesTr'
label_dir = '/home/ubuntu/liver/data/Task03_Liver_rs/labelsTr'
dataset = MedicalImageDataset(image_dir='/home/ubuntu/liver/data/Task03_Liver_rs/imagesTr', label_dir='/home/ubuntu/liver/data/Task03_Liver_rs/labelsTr', transform=transform)
image_filenames = sorted(os.listdir(image_dir))


train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# for i, (img, mask) in enumerate(train_dataset):
#     print(img.shape, mask.shape)
#     if i == 10:
#         break

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

for i, (img, mask) in enumerate(train_loader):
    print(img.shape, mask.shape)
    if i == 10:
        break







#
# class Segmenter(pl.LightningModule):
#
#     def __init__(self):
#         super().__init__()
#
#         self.model = UNet()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
#         self.loss_fn = torch.nn.CrossEntropyLoss()
#
#     def forward(self, data):
#         return self.model(data)
#
#     def training_step(self, batch, batch_idx):
#         img = batch['CT']['data']
#         mask = batch['Label']['data'][:, 0]
#         mask = mask.long()
#
#         y_hat = self(img)
#
#         loss = self.loss_fn(y_hat, mask)
#         self.log('train_loss', loss)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         img = batch['CT']['data']
#         mask = batch['Label']['data'][:, 0]
#         mask = mask.long()
#
#         y_hat = self(img)
#
#         loss = self.loss_fn(y_hat, mask)
#         self.log('val_loss', loss)
#         return loss
#
#     def configure_optimizers(self):
#         return [self.optimizer]



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
        mask = mask.long()

        y_hat = self(img)
        loss = self.loss_fn(y_hat, mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        mask = mask.long()

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



