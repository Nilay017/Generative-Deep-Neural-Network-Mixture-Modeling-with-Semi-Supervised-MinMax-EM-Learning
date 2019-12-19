import numpy as np
from data_processing import *

save_path = "./"
digits = np.array([0, 1, 2, 3, 4, 5, 6])
img_size = 28
num_images_per_cluster = 1000
seeds = [1.0, 2.0, 3.0, 4.0]


for seed in seeds:
	create_and_store_MNISTdataset(save_path, digits, img_size, num_images_per_cluster, seed)