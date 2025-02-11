import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
# from preprocess_dataset import preprocess_dataset_from_hf


def eda_diff_count(df: pd.DataFrame, symbol: str) -> None:
    diff = df["text_ko"].str.count(symbol) - df["text_en"].str.count(symbol)
    p1, p99 =np.percentile(diff, [1, 99])
    print(f"p1 : {p1}, p99 : {p99}")
    filter_diff = diff[(p1 <= diff) & (diff <= -1)]
    plt.hist(filter_diff, bins=100)
    plt.savefig(f"symbol_diff_{symbol}.png")
    counts = diff.value_counts().sort_index(ascending=False)
    for idx, val in counts.items():
        print(f"{idx:>5} : {val:>10}")

def checker_linebreak(df_join: pd.DataFrame) -> pd.Series:
    """
    resample 이 필요한 상황인지 아닌지 검사 : linebreak 숫자 차이가 지나치게 크면 resample
    return
        - True : resample 필요
        - False : resample 필요없음
    """
    linebreak_diff = df_join["text_ko"].str.count("\n") - df_join["text_en"].str.count("\n")
    return abs(linebreak_diff) > 3

def checker_backtick(df_join: pd.DataFrame) -> pd.Series:
    """
    resample 이 필요한 상황인지 아닌지 검사 : backtick 기호 개수가 다르면 resample
    return
        - True : resample 필요
        - False : resample 필요없음
    """
    backtick_diff = df_join["text_ko"].str.count("`") - df_join["text_en"].str.count("`")
    return backtick_diff != 0

def checker_all(df_join: pd.DataFrame) -> pd.Series:
    return checker_linebreak(df_join) | checker_backtick(df_join)

def append_resample_column(df_join: pd.DataFrame) -> pd.DataFrame:
    appended = df_join.copy()
    appended["resample"] = checker_all(df_join)
    return appended
    
# def resample_checker_code_translate(text_ko: str, text_en: str) -> bool:
#     """
#     resample 이 필요한 상황인지 아닌지 검사 : 주석 아닌 code 를 번역하는 경우 resample
#     return
#         - True : resample 필요
#         - False : resample 필요없음
#     """
#     if resample_checker_backtick(text_ko, text_en):
#         return False
    
#     def extract_code_block(text: str) -> list[str]:
#         """
#         ``` 으로 감싸진 substring 을 추출하는 함수
#         return
#             - list[str] : code block 으로 추정되는 substring 들
#         """
#         pattern = r"```(?:\w+)?\n(.*?)```"  # \w+ 대신 범용 패턴 사용
#         code_blocks = re.findall(pattern, text, re.DOTALL)
#         return code_blocks
    
#     def if_it_is_code_then_remove_comments(text, lang="auto"):
#         """
#         코드에서 주석을 제거하는 함수
        
#         Args:
#             text (str): 입력 코드
#             lang (str): 언어 (기본값 "auto" → 자동 감지)
        
#         Returns:
#             str: 주석이 제거된 코드
#         """
#         # 자동으로 언어 감지
#         if lang == "auto":
#             try:
#                 lexer = guess_lexer(text)
#                 lang = lexer.name.lower()
#             except ClassNotFound:
#                 print(text)
#                 return False  # 코드가 함수 종료

#         # 여러 언어의 주석 패턴 정의
#         comment_patterns = {
#             "python": [r"#.*"],  # Python 단일 라인 주석
#             "javascript": [r"//.*", r"/\*.*?\*/"],  # JS 단일/멀티라인 주석
#             "java": [r"//.*", r"/\*.*?\*/"],  # Java 단일/멀티라인 주석
#             "c": [r"//.*", r"/\*.*?\*/"],  # C, C++ 주석
#             "c++": [r"//.*", r"/\*.*?\*/"],  # C++ 주석
#             "html": [r"<!--.*?-->"],  # HTML/XML 주석
#             "xml": [r"<!--.*?-->"],  # XML 주석
#             "sql": [r"--.*", r"/\*.*?\*/"],  # SQL 단일/멀티라인 주석
#         }

#         # 주석 패턴에 따라 제거 수행
#         patterns = comment_patterns.get(lang, [])
#         for pattern in patterns:
#             text = re.sub(pattern, "", text, flags=re.DOTALL)

#         return text.strip()  # 앞뒤 공백 제거 후 반환
    


# if __name__ == "__main__":
#     pro500k = preprocess_dataset_from_hf("pro500k")
#     pro500k = append_resample_column(pro500k)
#     print(len(pro500k))
#     print(pro500k.head())
#     print(sum(pro500k["resample"]))

#     pro500k = pro500k[pro500k["resample"]]
#     print(len(pro500k))
    
#     # rea150k = preprocess_dataset_from_hf("reasoning150k")
    
#     # dpo100k = preprocess_dataset_from_hf("dpo100k")
