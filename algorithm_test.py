import numpy as np

test_data = np.load("./data/temp_save/test_data.npz")
test_descriptor = test_data["test_descriptor"]
for i in range(len(test_descriptor)):
    print(test_descriptor[i])
