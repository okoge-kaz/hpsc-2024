#!/bin/sh

module load hdf5-parallel/1.14.3/gcc11.4.1
export CPATH=$CPATH:/apps/t4/rhel9/free/hdf5-parallel/1.14.3/gcc11.4.1/include/

make 16_phdf5_write
mpirun -np 4 ./a.out

make 17_phdf5_read
mpirun -np 4 ./a.out
