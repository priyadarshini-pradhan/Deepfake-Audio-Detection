import numpy as np

y = np.load("Features/y.npy")

print("Total Samples:", len(y))
print("Real:", sum(y == 0))
print("Fake:", sum(y == 1))