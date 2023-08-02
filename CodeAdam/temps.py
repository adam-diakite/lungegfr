
def load_and_predict_quartiles(root_folder, output_folder, model):
    patient_info_list = []

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

            # Calculate mean between Q1 and Q3 of the scores for each patient
            q1 = np.percentile(predictions[:, 0], 25)
            q3 = np.percentile(predictions[:, 0], 75)
            mean_quartiles = (q1 + q3) / 2.0

            # Add patient information to the list
            patient_info = {
                "patient_folder": patient_folder,
                "mean_quartiles_score": mean_quartiles
            }
            patient_info_list.append(patient_info)

            print(f"Mean score between Q1 and Q3 for patient {patient_folder}: {mean_quartiles}")

    # Return patient information list
    return patient_info_list