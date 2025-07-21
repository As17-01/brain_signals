#!/bin/bash
#SBATCH --job-name=brain_states_bilstm_conv50_d2                # Название задачи 
#SBATCH --error=brain_states_bilstm_conv50_d2-%j.err                # Файл для вывода ошибок 
#SBATCH --output=brain_states_bilstm_conv50_d2-%j.log               # Файл для вывода результатов 
#SBATCH --time=96:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=0                            # Требуемое кол-во GPU

module load Python
source activate brain_states_env

poetry run python main.py model=bilstm_conv50_d2