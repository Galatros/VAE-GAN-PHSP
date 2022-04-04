#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J GAGA_ZT_job

## Liczba alokowanych węzłów
#SBATCH -N 1

## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1

## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=35GB

## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00 

## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgonwelo

## Specyfikacja partycji
#SBATCH -p plgrid-gpu-v100
#SBATCH --gres=gpu

## Plik ze standardowym wyjściem
#SBATCH --output="output_0.out"

## Plik ze standardowym wyjściem błędów
#SBATCH --error="error_0.err"


## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname

module load plgrid/tools/python/3.8

##python3 trainCGAN.py
python3 testCGAN.py
python3 Histogramy.py



