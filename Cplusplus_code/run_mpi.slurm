#!/bin/bash

#SBATCH -J TSQR_MPI
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --partition=24core

## Compilation
mpiicpc ./generate_matrix.cpp -o generate_matrix
mpiicpc ./qr_mpi.cpp -o qr_mpi -mkl

## Generate matrix ( rows, columns, blocks ) and save it in file 'input_matrix.dat'
./generate_matrix 204800 200 16 > input_matrix.dat

## Print input matrix, QR decomposition, and compute time
#mpiexec -n 4 ./qr_mpi -file input_matrix.dat

## Print compute time only - for benchmark purposes
mpiexec -n 16 ./qr_mpi -file input_matrix.dat -silent

## Delete files
rm generate_matrix input_matrix.dat qr_mpi
