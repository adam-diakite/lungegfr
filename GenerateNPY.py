import os
import numpy as np
from PIL import Image


def normalization_img(image):
    return (image - np.mean(image)) / np.std(image)


label = np.load('/home/adamdiakite/Documents/lungegfr-master/Sampledata/label.npy')

ct_files = os.listdir('/home/adamdiakite/Documents/lungegfr-master/Sampledata/Image/CT')[2:]
num_files = len(ct_files)
cts = np.empty((num_files, 64, 64), dtype=np.float64)
pets = np.empty((num_files, 64, 64), dtype=np.float64)
fuses = np.empty((num_files, 64, 64), dtype=np.float64)

for i, file in enumerate(ct_files):

    ct_path = os.path.join('/home/adamdiakite/Documents/lungegfr-master/Sampledata/Image/CT/', file)
    pet_path = os.path.join('/home/adamdiakite/Documents/lungegfr-master/Sampledata/Image/PET/', file)

    ct_image = np.array(Image.open(ct_path), dtype=np.float64)
    pet_image = np.array(Image.open(pet_path), dtype=np.float64)

    pet_image = normalization_img(np.array(Image.fromarray(pet_image).resize((64, 64), Image.NEAREST)))
    ct_image = normalization_img(np.array(Image.fromarray(ct_image).resize((64, 64), Image.NEAREST)))

    pet_image = (pet_image - np.mean(pet_image)) / np.std(pet_image)
    ct_image = (ct_image - np.mean(ct_image)) / np.std(ct_image)
    fuse_image = pet_image + ct_image
    fuse_image = (fuse_image - np.mean(fuse_image)) / np.std(fuse_image)

    cts[i] = ct_image
    pets[i] = pet_image
    fuses[i] = fuse_image

label = label[:num_files]

cts = np.transpose(cts, (0, 2, 1))
pets = np.transpose(pets, (0, 2, 1))
fuses = np.transpose(fuses, (0, 2, 1))

np.save('/home/adamdiakite/Documents/lungegfr-master/Sampledata/xpsamplect.npy', cts)
np.save('/home/adamdiakite/Documents/lungegfr-master/Sampledata/xpsamplepet.npy', pets)
np.save('/home/adamdiakite/Documents/lungegfr-master/Sampledata/xpsamplefuse.npy', fuses)