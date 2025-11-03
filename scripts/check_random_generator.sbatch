#!/bin/bash
#SBATCH --job-name=check_random_generator                # Название задачи 
#SBATCH --error=check_random_generator-%j.err                # Файл для вывода ошибок 
#SBATCH --output=check_random_generator-%j.log               # Файл для вывода результатов 
#SBATCH --time=00:05:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=0                            # Требуемое кол-во GPU

module load Python
source activate brain_states_env

poetry run python check_random.py model=gru_default