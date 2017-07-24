from scipy.ndimage import label
import numpy as np

a = np.array([[0, 0, 1, 1, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [1, 1, 0, 0, 1, 0],
              [0, 0, 0, 1, 0, 0]])
labeled_array, num_features = label(a)
print("dddd")
