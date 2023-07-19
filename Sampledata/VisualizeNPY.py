import numpy as np
import matplotlib.pyplot as plt
#
# Load NPY file
data = np.load('/home/lito/PycharmProjects/lungegfr/Sampledata/xpsamplefuse.npy')

# Determine the number of images and their shape
num_images, height, width = data.shape

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10, 4))

# Iterate over the images and display them
for i in range(num_images):
    axes[i].imshow(data[i], cmap='gray')
    axes[i].axis('off')  # Hide the axes
    axes[i].set_title(f'Image {i+1}')

plt.tight_layout()
plt.show()

