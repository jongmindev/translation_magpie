# Workflow

1. batch file 만들기 : `resample` 폴더에서
    - 처음  
    `python make_batch.py --source hf --num_shards 4 --output_dir batch-0 --force_overwrite`
    - 2회차 이후  
    `python make_batch.py --source results-{N-1} --num_shards 4 --output_dir batch-{N} --force_overwrite`
2. vllm 실행 : `resample` 폴더에서
    - `bash scripts/submit.sh <round_num> <total_split_num>`