#!/bin/bash
python3 get_data.py


python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_5_seed_0.0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 0 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_5_seed_1.0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 1 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_5_seed_2.0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 2 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_5_seed_3.0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 3 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_5_seed_4.0.pkl -out ./Output_data_processing -g 7 -num_clusters 5 -seed 4

python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_7_seed_0.0.pkl -out ./Output_data_processing -g 7 -num_clusters 7 -seed 0 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_7_seed_1.0.pkl -out ./Output_data_processing -g 7 -num_clusters 7 -seed 1 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_7_seed_2.0.pkl -out ./Output_data_processing -g 7 -num_clusters 7 -seed 2 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_7_seed_3.0.pkl -out ./Output_data_processing -g 7 -num_clusters 7 -seed 3 
python3 data_processing.py -shuffle True -use_cuda True -in ./CIFAR10_numperdigit_1000_numdigits_7_seed_4.0.pkl -out ./Output_data_processing -g 7 -num_clusters 7 -seed 4
