from __future__ import print_function
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

folder = "/media/adamdiakite/LaCie/CT-TEP_Data/NPY"
results ="/media/adamdiakite/LaCie/CT-TEP_Data/Results"
modelpetct1 = load_model('/home/lito/PycharmProjects/lungegfr/model/LungEGFR.hdf5')  # ,weightspatient2-improvement-40-0.67
modelpetct1.summary()


def load_and_predict(root_folder, output_folder, model):
    for root, dirs, _ in os.walk(root_folder):
        if "ct.npy" in dirs and "pet.npy" in dirs and "fuse.npy" in dirs and "label.npy" in dirs:
            print(f"Processing folder: {root}")

            ct_npy = np.load(os.path.join(root, "ct.npy"))
            pet_npy = np.load(os.path.join(root, "pet.npy"))
            fuse_npy = np.load(os.path.join(root, "fuse.npy"))
            label_npy = np.load(os.path.join(root, "label.npy"))

            ct_data = np.asarray(ct_npy, dtype="float32")
            pet_data = np.asarray(pet_npy, dtype="float32")
            fuse_data = np.asarray(fuse_npy, dtype="float32")
            label_data = np.asarray(label_npy, dtype="float32")

            ct_data = np.expand_dims(ct_data, axis=3)
            pet_data = np.expand_dims(pet_data, axis=3)
            fuse_data = np.expand_dims(fuse_data, axis=3)

            x_data = np.concatenate((pet_data, ct_data, fuse_data), axis=3)
            predictions = model.predict(x_data, verbose=1)

            # Save predictions
            patient_name = os.path.basename(root)
            patient_output_folder = os.path.join(output_folder, patient_name)
            os.makedirs(patient_output_folder, exist_ok=True)
            np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

            print(f"Predictions for folder {patient_name} saved successfully.")

load_and_predict(folder, results, modelpetct1)
load_and_predict(folder, results, modelpetct1)