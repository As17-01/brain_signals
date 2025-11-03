#!/bin/bash
#SBATCH --job-name=brain_states_liquid_lag1                # Название задачи 
#SBATCH --error=brain_states_liquid_lag1-%j.err                # Файл для вывода ошибок 
#SBATCH --output=brain_states_liquid_lag1-%j.log               # Файл для вывода результатов 
#SBATCH --time=128:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=2                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate brain_states_env

poetry run python main.py model=liquid_lag1