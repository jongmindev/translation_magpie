import os
from glob import glob
import numpy as np
import pandas as pd
import datasets


def dataset_hf_to_dataset_join(dataset_name: str) -> pd.DataFrame:
    """
    english dataset 과 korean dataset 을 huggingface 로부터 load 한뒤 join 하는 함수
    return
        - (SFT) pd.DataFrame [uuid, task_category, / instruction_ko, response_ko,  / instruction_en, response_en]
        - (DPO) pd.DataFrame [uuid, task_category, / instruction_ko, chosen_ko, rejected_ko, / instruction_en, chosen_en, rejected_en]
    """
    # print(f"[Dataset name] : {dataset_name}")
    if dataset_name == "pro500k":
        dataset_ko_name = "youjunhyeok/Magpie-Llama-3.1-Pro-500K-Filtered-ko"
        dataset_en_name = "Magpie-Align/Magpie-Llama-3.1-Pro-500K-Filtered"
        dataset_dict_ko = datasets.load_dataset(dataset_ko_name)
        dataset_dict_en = datasets.load_dataset(dataset_en_name)
        dataset_dict_ko = dataset_dict_ko.select_columns(['uuid', 'instruction', 'response'])
        dataset_dict_en = dataset_dict_en.select_columns(['uuid', 'task_category', 'instruction', 'response'])

    elif dataset_name == "reasoning150k":
        dataset_ko_name = "GentleDrum/Magpie-Reasoning-V1-150K-ko"
        dataset_en_name = "Magpie-Align/Magpie-Reasoning-V1-150K"
        dataset_dict_ko = datasets.load_dataset(dataset_ko_name)
        dataset_dict_en = datasets.load_dataset(dataset_en_name)
        dataset_dict_ko = dataset_dict_ko.select_columns(['uuid', 'instruction', 'response'])
        dataset_dict_en = dataset_dict_en.select_columns(['uuid', 'task_category', 'instruction', 'response'])

    elif dataset_name == "dpo100k":
        dataset_ko_name = "youjunhyeok/Magpie-Llama-3.1-Pro-DPO-100K-v0.1-ko"
        dataset_en_name = "Magpie-Align/Magpie-Llama-3.1-Pro-DPO-100K-v0.1"
        dataset_dict_ko = datasets.load_dataset(dataset_ko_name)
        dataset_dict_en = datasets.load_dataset(dataset_en_name)
        dataset_dict_ko = dataset_dict_ko.select_columns(['uuid', 'instruction', 'chosen', 'rejected'])
        dataset_dict_en = dataset_dict_en.select_columns(['uuid', 'task_category', 'instruction', 'chosen', 'rejected'])
        def format_chosen_rejected(row):
            assert type(row["chosen"]) == list
            assert len(row["chosen"]) == 2
            assert type(row["rejected"]) == list
            assert len(row["rejected"]) == 2
            # assert row["chosen"][0]["content"] == row["rejected"][0]["content"] == row["instruction"]
            chosen = row["chosen"][1]["content"]
            rejected = row["rejected"][1]["content"]
            row["chosen"] = chosen
            row["rejected"] = rejected
            return row
        dataset_dict_ko = dataset_dict_ko.map(format_chosen_rejected, num_proc=16)
        dataset_dict_en = dataset_dict_en.map(format_chosen_rejected, num_proc=16)
    else:
        raise ValueError(f"Invalid dataset_name : {dataset_name}")
    
    # print(dataset_dict_ko)
    # print(dataset_dict_en)
    
    dataset_ko = datasets.concatenate_datasets([
        split for split in dataset_dict_ko.values()
    ])
    dataset_en = datasets.concatenate_datasets([
        split for split in dataset_dict_en.values()
    ])
    # print(len(dataset_ko), len(dataset_en))

    dataset_join = pd.merge(
        dataset_ko.to_pandas(), dataset_en.to_pandas(), 
        on="uuid", how="inner", suffixes=("_ko", "_en")
    )
    dataset_join['task_category'] = dataset_join['task_category'].fillna("Unknown")
    # print(len(dataset_join))
    # print(dataset_join.columns)
    # print(dataset_join.head())

    return dataset_join

def flatten_dataset(dataset_join: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    join 된 dataset 을 flatten 하는 함수
    return
        - (SFT, DPO 동일) pd.DataFrame [dataset_name, column_name, uuid, task_category, / text_ko, text_en]
    """
    def helper(dataset_join: pd.DataFrame, dataset_name: str, column_name: str) -> pd.DataFrame:
        assert dataset_name in ["pro500k", "reasoning150k", "dpo100k"], f"Invalid dataset_name : {dataset_name}"
        assert column_name in ["instruction", "response", "chosen", "rejected"], f"Invalid column_name : {column_name}"

        df = dataset_join[['uuid', 'task_category', f'{column_name}_ko', f'{column_name}_en']]
        df = df.rename(columns={f'{column_name}_ko': f'text_ko', f'{column_name}_en': f'text_en'})
        df.insert(0, 'dataset_name', dataset_name)
        df.insert(1, 'column_name', column_name)
        return df
    
    if dataset_name == "pro500k":
        column_names = ["instruction", "response"]
    elif dataset_name == "reasoning150k":
        column_names = ["instruction", "response"]
    elif dataset_name == "dpo100k":
        column_names = ["instruction", "chosen", "rejected"]
    else:
        raise ValueError(f"Invalid dataset_name : {dataset_name}")
    
    flattened_by_column = [helper(dataset_join, dataset_name, column_name) for column_name in column_names]
    flattened = pd.concat(flattened_by_column, ignore_index=True)
    return flattened

def preprocess_dataset_from_hf(dataset_name: str) -> pd.DataFrame:
    assert dataset_name in ["pro500k", "reasoning150k", "dpo100k"], f"Invalid dataset_name : {dataset_name}"
    dataset_join = dataset_hf_to_dataset_join(dataset_name)
    flattened = flatten_dataset(dataset_join, dataset_name)
    return flattened

###############

def extract_content(row: dict) -> str:
    if pd.notna(row["error"]):      # error 컬럼이 NaN 이 아닌 경우 (오류 메시지가 존재하면)
        return np.nan               # content 를 NaN 으로 return

    choices = row["response"]["body"]["choices"]
    assert type(choices) == list
    assert len(choices) == 1, f"Error: choices length is {len(choices)}"
    return choices[0]["message"]["content"]

def parse_custom_id(row: dict) -> pd.Series:
    metadata = row["custom_id"].split(",")
    return pd.Series({
        "uuid": metadata[0],
        "dataset_name": metadata[1],
        "column_name": metadata[2]
    })

def make_df_ko_from_vllm_response(output: pd.DataFrame) -> pd.DataFrame:
    output["text_ko"] = output.apply(extract_content, axis=1)
    output[["uuid", "dataset_name", "column_name"]] = output.apply(parse_custom_id, axis=1)
    output = output[["dataset_name", "column_name", "uuid", "text_ko"]]
    return output

def preprocess_dataset_from_vllm(vllm_dir: str) -> pd.DataFrame:
    # prepare df_ko
    assert os.path.exists(vllm_dir), f"File not found : {vllm_dir}"
    vllm_pathes = glob(f"{vllm_dir}/split-*.jsonl")
    # vllm_pathes = glob(f"{vllm_dir}/split-0.jsonl")
    assert len(vllm_pathes) > 0, f"No file in this directory : {vllm_dir}"
    outputs = [pd.read_json(path, lines=True) for path in vllm_pathes]
    output = pd.concat(outputs, ignore_index=True)
    df_ko = make_df_ko_from_vllm_response(output)
    # print(df_ko.columns)

    # prepare df_en
    dataset_names = ["pro500k", "reasoning150k", "dpo100k"]
    dfs_en = {
        dataset_name: preprocess_dataset_from_hf(dataset_name) for dataset_name in dataset_names
    }
    df_en = pd.concat(dfs_en.values(), ignore_index=True)
    df_en = df_en[["dataset_name", "column_name", "uuid", "task_category", "text_en"]]

    # prepare df_join
    df_join = df_ko.merge(df_en, on=["dataset_name", "column_name", "uuid"], how="inner")
    # print(len(df_ko), len(df_en), len(df_join))
    return df_join


if __name__ == "__main__":

    def find_non_string_elements(df, max_results=10):
        non_string_elements = []

        for col in df.columns:
            for idx, value in df[col].items():
                if not isinstance(value, str):
                    non_string_elements.append((idx, col, value))  # (행 인덱스, 컬럼명, 값)
                    if len(non_string_elements) >= max_results:
                        return non_string_elements  # 최대 개수 도달 시 반환

        return non_string_elements  # 결과 반환

    pro500k = dataset_hf_to_dataset_join("pro500k")
    pro500k = flatten_dataset(pro500k, "pro500k")
    print(pro500k.columns)
    # print(pro500k.head())

    # diff = pro500k['text_ko'].str.count('\n') - pro500k['text_en'].str.count('\n')
    # # print(diff)
    # resample1 = abs(diff) > 3
    # print(resample1)
    # resample2 = abs(diff) > 3
    # resample_all = resample1 | resample2
    # print(resample_all)
    # print(sum(resample1), sum(resample2), sum(resample_all))
    # # print(type(resample))
    # # print(sum(resample))
    # # print(resample)
    
    # # rea150k = dataset_hf_to_dataset_join("reasoning150k")
    # # rea150k = flatten_dataset(rea150k, "reasoning150k")
    
    # # dpo100k = dataset_hf_to_dataset_join("dpo100k")
    # # dpo100k = flatten_dataset(dpo100k, "dpo100k")

    df_ko = preprocess_dataset_from_vllm("/home/n6/jongmin/translation/resample2/outputs")
    # print(df_ko)
    print(df_ko.columns)
    # print(df_ko.head())

    