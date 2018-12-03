#!/bin/bash

#SBATCH -J TSQR
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 00:20:00
#SBATCH --partition=gpu

../../../legion_11_2018/language/regent.py TSQR_main_complete_QR.rg -fcuda 0 -fflow 1 

#-ll:cpu 8 -ll:csize 16000 -lg:prof 1 -lg:prof_logfile prof_%.gz
#../../legion_11_2018/tools/legion_prof.py prof_*.gz

