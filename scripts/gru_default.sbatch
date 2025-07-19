#!/bin/bash
#SBATCH --job-name=brain_states_gru_default                # Название задачи 
#SBATCH --error=brain_states_gru_default-%j.err                # Файл для вывода ошибок 
#SBATCH --output=brain_states_gru_default-%j.log               # Файл для вывода результатов 
#SBATCH --time=96:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python
source activate brain_states_env

poetry run python main.py model=gru_default