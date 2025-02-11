#!/bin/bash
#SBATCH --job-name=trans
#SBATCH --partition=PB
#SBATCH --nodelist=b3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --output=/home/n6/jongmin/translation/resample2/logs/trans/%x-%j.out
#SBATCH --error=/home/n6/jongmin/translation/resample2/logs/trans/%x-%j.err

source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"
conda activate trans
cd /home/n6/jongmin/translation/resample2

input_jsonl_path=$1
output_jsonl_path=$2

python \
    -m vllm.entrypoints.openai.run_batch \
    -i $input_jsonl_path \
    -o $output_jsonl_path \
    --model nayohan/llama3-instrucTrans-enko-8b