#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=0:01:00
#$ -o outputs/$JOB_ID
#$ -e outputs/$JOB_ID
./a.out
