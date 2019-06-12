#!/bin/bash

#SBATCH -J TSQR
#SBATCH -N 4
#SBATCH -n 32
#SBATCH -t 00:20:00
#SBATCH --partition=32core

#$LEGION_DIR/language/regent.py TSQR_main_complete_QR.rg -ll:cpu 16 -ll:csize 30000 -fcuda 0 
#-lg:prof 1 -lg:prof_logfile prof_%.gz
#$LEGION_CR_DIR/tools/legion_prof.py prof_*.gz
#$LEGION_CR_DIR/language/regent.py TSQR_main_complete_QR.rg -ll:cpu 2 -ll:csize 24000 -fcuda 0 
#-lg:spy -logfile spy_%.log
#$LEGION_DIR/tools/legion_spy.py -dez spy_*.log
LAUNCHER='mpirun --bind-to none -np 4 -npernode 1' $LEGION_DIR/language/regent.py TSQR_main_complete_QR.rg -fcuda 0 -ll:cpu 16 -ll:csize 24000 

#-lg:prof 8 -lg:prof_logfile prof_%.gz
#$LEGION_DIR/tools/legion_prof.py prof_*.gz
#-lg:spy -logfile spy_%.log
#$LEGION_DIR/tools/legion_spy.py -dez spy_*.log

