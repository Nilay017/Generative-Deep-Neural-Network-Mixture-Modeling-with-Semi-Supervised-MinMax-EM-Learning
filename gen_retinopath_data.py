from data_processing import *

for seed in [0.0, 1.0, 2.0, 3.0, 4.0]:
	create_and_store_DiabeticRetinopathydataset('./Diabetic_retinopathy_Data',\
	 '../Diabetic_retinopathy', seed=seed)

