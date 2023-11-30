import os
import numpy as np
import matplotlib.pyplot as plt


data = np.load('/media/lito/LaCie/CT-TEP_ICI/NPY/P_AAAA1139/fuse.npy')

# Determine the number of images and their shape
num_images, height, width = data.shape

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 4))

# Iterate over the images and display them
for i in range(num_images):
    axes[i].imshow(data[i], cmap='gray')
    axes[i].axis('off')  # Hide the axes
    axes[i].set_title(i + 1)

plt.tight_layout()
plt.show()
