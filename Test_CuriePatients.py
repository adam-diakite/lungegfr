from __future__ import print_function
import numpy as np
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

folder = "/media/adamdiakite/LaCie/NPY"
results ="/media/adamdiakite/LaCie/Results"
modelpetct1 = load_model('/home/adamdiakite/Documents/lungegfr-master/model/LungEGFR.hdf5')  # ,weightspatient2-improvement-40-0.67
modelpetct1.summary()


def load_and_predict(root_folder, output_folder, model):
    for patient_folder in os.listdir(root_folder):
        patient_folder_path = os.path.join(root_folder, patient_folder)

        if os.path.isdir(patient_folder_path):
            print(f"Processing patient folder: {patient_folder}")

            ct_npy_path = os.path.join(patient_folder_path, "ct.npy")
            pet_npy_path = os.path.join(patient_folder_path, "pet.npy")
            fuse_npy_path = os.path.join(patient_folder_path, "fuse.npy")
            label_npy_path = os.path.join(patient_folder_path, "label.npy")

            if not (os.path.exists(ct_npy_path) and os.path.exists(pet_npy_path)
                    and os.path.exists(fuse_npy_path) and os.path.exists(label_npy_path)):
                print(f"Skipping patient folder {patient_folder}. NPY files are missing.")
                continue

            ct_npy = np.load(ct_npy_path)
            pet_npy = np.load(pet_npy_path)
            fuse_npy = np.load(fuse_npy_path)
            label_npy = np.load(label_npy_path)

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
            patient_output_folder = os.path.join(output_folder, patient_folder)
            os.makedirs(patient_output_folder, exist_ok=True)
            np.savetxt(os.path.join(patient_output_folder, 'predict.txt'), predictions)

            print(f"Predictions for patient folder {patient_folder} saved successfully.")

load_and_predict(folder, results, modelpetct1)
