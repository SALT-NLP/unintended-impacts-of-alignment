from datasets import load_dataset
from typing import Tuple
from langdetect import detect_langs
from langdetect import DetectorFactory
from ftlangdetect import detect

from tqdm import tqdm

import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="LangId Ultrachat dataset")

    parser.add_argument(
        "--output_file",
        type=str,
        default="output.csv",
        help="The file to output results to",
    )

    return parser.parse_args()

# For reproducibility with langdetect
DetectorFactory.seed = 0

def get_lang_probs(text:str) -> Tuple[float, float]:
    ld_lang, ld_prob = "unknown", 0
    try:
        langdetect_output = detect_langs(text)
        langdetect = max(langdetect_output, key=lambda x: x.prob)
        ld_lang = langdetect.lang
        ld_prob = langdetect.prob
    except:
        pass
    fasttext_output = detect(text=text.replace('\n',''), low_memory=False)

    ft_lang = fasttext_output['lang']
    ft_prob = fasttext_output['score']

    return (ld_lang, ld_prob), (ft_lang, ft_prob)

def process_tulu(output_file):

    dataset = load_dataset("allenai/tulu-v2-sft-mixture")

    splits = ["train"]

    output = {
        "text": [],
        "role": [],
        "agreed_lang_text": [],
        "langid_lang_text": [],
        "langid_prob_text": [],
        "fasttext_lang_text": [],
        "fasttext_prob_text": [],
        "id": [],
        "turn": [],
        "dataset": []
    }

    for split in splits:
        for example in tqdm(dataset[split]):
            messages = example["messages"]

            for i, message in enumerate(messages):
                (ld_lang, ld_prob), (ft_lang, ft_prob) = get_lang_probs(message["content"])

                output["text"].append(message["content"])
                output["role"].append(message["role"])
                if ld_lang == ft_lang:
                    output["agreed_lang_text"].append(ld_lang)
                else:
                    output["agreed_lang_text"].append("unknown")
                output["langid_lang_text"].append(ld_lang)
                output["langid_prob_text"].append(ld_prob)
                output["fasttext_lang_text"].append(ft_lang)
                output["fasttext_prob_text"].append(ft_prob)
                output["id"].append(example["id"])
                output["turn"].append(i)
                output["dataset"].append(example["dataset"])

    df = pd.DataFrame(output)

    # Check that the output directory exists
    if os.path.dirname(output_file) != "" and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    df.to_csv(output_file, index=False)

def main():
    args = parse_args()
    process_tulu(args.output_file)

if __name__ == "__main__":
    main()

# Example usage:
# python code/langid-tulu-sft.py --output_file=./outputs/tulu-sft-langid/langid.csv