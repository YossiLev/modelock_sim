#!/bin/bash
set -e  # Script will stop on the first error

nvcc -c -Xcompiler -fPIC ./cfuncs/diode_cavity.cu -o ./cfuncs/diode_cavity.o
echo "cuda compilation done"
gcc -c -fPIC -DUSE_CUDA_CODE ./cfuncs/diode_actions.c -o ./cfuncs/diode_actions.o 
echo "C compilation done"
nvcc -shared -o ./cfuncs/libs/libdiode.so ./cfuncs/diode_actions.o ./cfuncs/diode_cavity.o -lcufft -lcudart
echo "Libeary ready"