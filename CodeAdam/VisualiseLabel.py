import numpy as np

# Replace 'file_path.npy' with the path to your NPY file
file_path = '/media/adamdiakite/LaCie/CT-TEP_Data/PNG/2-21-0038/label.npy'

# Load the NPY file
data = np.load(file_path)

# Now you can work with the data, for example, print its content
print(data)