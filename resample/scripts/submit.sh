#!/bin/bash

function submit(){
    
    round_num=$1
    split_num=$2

    jn=b${round_num}/s${split_num}
    echo "Submit translation - jobname : $jn"

    input_jsonl_path="${HOME}/translation/resample2/batch-${round_num}/split-${split_num}.jsonl"

    if [ -f $input_jsonl_path ]; then
        echo "input jsonl file : $input_jsonl_path"
    else
        echo "[ERROR] File not found: $input_jsonl_path"
    fi

    output_jsonl_path="$HOME/translation/resample2/results-${round_num}/split-${split_num}.jsonl"
    # echo "output jsonl file : $output_jsonl_path"
    sbatch -J $jn scripts/run.sh $input_jsonl_path $output_jsonl_path
    echo
}

# 최소 2개의 인자가 없으면 오류 발생
if [ $# -lt 2 ]; then
    echo "[ERROR] Not enough arguments : bash submit.sh <round_num> <total_split_num>"
    exit 1
fi

round_num=$1
total_split_num=$2

# output 폴더 관련 처리 : 없으면 만들고, 있으면 비어있는 경우에만 이후 진행 (덮어쓰기 방지)
output_dir="$HOME/translation/resample2/results-$round_num"
if [ ! -d $output_dir ]; then
    echo "Output directory not exists, so make : $output_dir"
    mkdir -p $output_dir
elif [ "$(ls -A "$output_dir")" ]; then
    echo "[Error] Output directory is not empty! : $output_dir"
    exit 1
fi

# echo "Submit"

for split_num in $(seq 1 $total_split_num); do
    submit $round_num $split_num
done