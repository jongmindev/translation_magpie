import os
from argparse import ArgumentParser
import pandas as pd
import datasets
from preprocess_dataset import (
    preprocess_dataset_from_hf,
    preprocess_dataset_from_vllm
)
from filter_to_resample import append_resample_column


def make_batch(row):
    output = {}
    output["custom_id"] = f"{row['uuid']},{row['dataset_name']},{row['column_name']}"
    output["method"] = "POST"
    output["url"] = "/v1/chat/completions"
    output["body"] = {
        "model": "nayohan/llama3-instrucTrans-enko-8b",
        "messages": [
            {"role": "system", "content": "당신은 번역기 입니다. 영어를 한국어로 번역하세요."},
            {"role": "user", "content": row['text_en']}
        ],
        "max_completion_tokens": 4096
    }
    return output

# 각 컬럼별 빈도수 및 비율 계산
def frequency_stats(df, column):
    freq = df[column].value_counts()  # 빈도수
    ratio = df[column].value_counts(normalize=True) * 100  # 비율 (%)
    
    stats_df = pd.DataFrame({"count": freq, "percentage": ratio})
    return stats_df

def apply_make_batch(
        source: str,
        num_shards: int, 
        output_dir: str = "./batch-0", 
        verbose: int = 1,
        overwrite: bool = False
    ) -> datasets.Dataset:

    if overwrite:
        os.makedirs(output_dir, exist_ok=True)
    else:
        if os.path.exists(output_dir):
            assert not any(os.scandir(output_dir)), f"Output directory already exists and is not empty : {output_dir}"
        else:
            os.makedirs(output_dir, exist_ok=False)
    
    if source == "hf":
        dataset_names = ["pro500k", "reasoning150k", "dpo100k"]
        magpie_datasets_to_resample = {
            dataset_name: preprocess_dataset_from_hf(dataset_name) for dataset_name in dataset_names
        }
        magpie_datasets_to_resample = {
            dataset_name: append_resample_column(dataset) for dataset_name, dataset in magpie_datasets_to_resample.items()
        }
        magpie_datasets_to_resample = {
            dataset_name: dataset[dataset["resample"]] for dataset_name, dataset in magpie_datasets_to_resample.items()
        }
        magpie_dataset = pd.concat(magpie_datasets_to_resample.values(), ignore_index=True)
    
        if verbose > 0:
            print(frequency_stats(magpie_dataset, "dataset_name"))
            print(frequency_stats(magpie_dataset, "column_name"))
            print(frequency_stats(magpie_dataset, "task_category"))
    else:
        magpie_dataset_to_resample = preprocess_dataset_from_vllm(source) 
        magpie_dataset_to_resample = append_resample_column(magpie_dataset_to_resample)
        print(f"Source : vLLM | before : {len(magpie_dataset_to_resample)}")
        magpie_dataset = magpie_dataset_to_resample[magpie_dataset_to_resample["resample"]]
        print(f"Source : vLLM |  after : {len(magpie_dataset)}")
    
    print("[Batching] Dataset instanceds to resample are filtered!")

    # print("uuid :\n", magpie_dataset['uuid'].describe())

    # magpie_dataset = datasets.Dataset.from_pandas(magpie_dataset).select(range(64))
    magpie_dataset = datasets.Dataset.from_pandas(magpie_dataset)
    magpie_dataset = magpie_dataset.cast(magpie_dataset.features)
    columns_to_remove = magpie_dataset.column_names
    magpie_dataset = magpie_dataset.map(
        lambda row: make_batch(row), 
        with_indices=False, 
        num_proc=16,
        remove_columns=columns_to_remove
    ).shuffle(seed=42)
    for i in range(num_shards): 
        shard = magpie_dataset.shard(num_shards=num_shards, index=i)
        shard.to_json(f"{output_dir}/split-{i+1}.jsonl", force_ascii=False)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("-f", "--force_overwrite", action="store_true")
    args = parser.parse_args()

    if args.source != "hf":
        assert os.path.exists(args.source)
    
    # apply_make_batch(source=args.source, num_shards=2, output_dir="./sample", overwrite=args.force_overwrite)
    apply_make_batch(
        source=args.source, 
        num_shards=args.num_shards, 
        output_dir=args.output_dir, 
        verbose=args.verbose,
        overwrite=args.force_overwrite
    )
    