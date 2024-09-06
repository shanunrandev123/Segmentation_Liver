import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from celluloid import Camera
from IPython.display import HTML
from pathlib import Path
import warnings


root = Path('/home/ubuntu/liver/data/Task03_Liver_rs/imagesTr')

label = Path('/home/ubuntu/liver/data/Task03_Liver_rs/labelsTr')

def change_img_to_label_path(path):

    parts = list(path.parts)
    parts[parts.index('imagesTr')] = 'labelsTr'
    return Path(*parts)

sample_path = list(root.glob('liver*'))[4]

print(sample_path)

sample_label_path = change_img_to_label_path(sample_path)

print(sample_label_path)


ct = nib.load(sample_path).get_fdata()
mask = nib.load(sample_label_path).get_fdata().astype(int)





fig = plt.figure(figsize=(10, 10))

camera = Camera(fig)

for i in range(ct.shape[2]):
    plt.imshow(ct[:, :, i],cmap='bone')

    masked_array = np.ma.masked_where(mask[:, :, i] == 0, mask[:, :, i])
    plt.imshow(masked_array, alpha=0.5)
    camera.snap()

animation = camera.animate()
plt.show()
