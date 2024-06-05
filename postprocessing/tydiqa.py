
""" Based on official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import argparse
import os
from tqdm import tqdm
from functools import lru_cache
from qa_metrics.cfm import CFMatcher

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def remove_extraneous_generations(text):
        return text.split("\n\n")[0].strip()

    return white_space_fix(remove_articles(remove_punc(lower(remove_extraneous_generations(s)))))


@lru_cache(maxsize=1)
def get_cfm():
    return CFMatcher()

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate_list(dataset, predictions):
    f1 = exact_match = cfm_score = cfm_match = total = 0
    f1_list = []
    exact_match_list = []
    cfm_list = []
    cfm_match_list = []
    cfm = get_cfm()
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                prediction = prediction.split("\n\n")[0].strip() # Remove extraneous generations (NEW)
                curr_exact_match_score = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                curr_f1_score = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                
                cfm_scores = []
                match_results = []
                for gt in ground_truths:
                    cfm_scores.append(cfm.get_scores(gt, prediction, qa['question'])[gt][prediction])
                    match_results.append(1 if cfm.cf_match(gt, prediction, qa['question']) else 0)
                
                curr_cfm_score = max(cfm_scores)
                curr_match_result = max(match_results)

                exact_match += curr_exact_match_score
                f1 += curr_f1_score
                cfm_score += curr_cfm_score
                cfm_match += curr_match_result

                exact_match_list.append(100.0 * curr_exact_match_score)
                f1_list.append(100.0 * curr_f1_score)
                cfm_list.append(100.0 * curr_cfm_score) 
                cfm_match_list.append(100.0 * curr_match_result)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    cfm_score = 100.0 * cfm_score / total
    cfm_match = 100.0 * cfm_match / total

    return {'exact_match': exact_match, 'f1': f1, 'exact_match_list': exact_match_list, 'f1_list': f1_list, 'cfm_score': cfm_score, 'cfm_match': cfm_match, 'cfm_list': cfm_list, 'cfm_match_list': cfm_match_list}

def parse_args():
    parser = argparse.ArgumentParser(description="Compute scores for TyDi QA")

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory containing model outputs",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/4_tydiqa/goldp.csv",
        help="Output file to write results to",
    )
    parser.add_argument(
        "--gold_dir",
        type=str,
        default="./data/tydiqa/tydiqa-goldp-v1.1-dev/",
        help="Directory containing tydiqa goldp solutions in all languages",
    )
    parser.add_argument(
        "--output_list",
        action="store_true",
        help="Output all scores for each example in the output file",
    )

    args = parser.parse_args()

    return args

def compute_scores(input_file, gold_file):
    expected_version = 'TyDiQA-GoldP-1.1-for-SQuAD-1.1'
    with open(gold_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(input_file) as prediction_file:
        predictions = json.load(prediction_file)
    return evaluate_list(dataset, predictions)

def compute_scores_dir(input_dir, output_file, gold_dir, output_list=False):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as f:
        f.write("model,language,decoding_strat,num_examples,exact_match,f1,cfm_score,cfm_match\n")

    languages = ["arabic", "bengali", "english", "finnish", "indonesian", "korean", "russian", "swahili", "telugu"]
    for model in tqdm(os.listdir(input_dir)):
        if os.path.isfile(os.path.join(input_dir, model)): continue
        for decoding_strat in os.listdir(os.path.join(input_dir, model)):
            if os.path.isfile(os.path.join(input_dir, model, decoding_strat)): continue
            for num_examples in os.listdir(os.path.join(input_dir, model, decoding_strat)):
                if os.path.isfile(os.path.join(input_dir, model, decoding_strat, num_examples)): continue
                for lang in languages:
                    input_file = os.path.join(input_dir, model, decoding_strat, num_examples, lang + ".json")
                    gold_file = os.path.join(gold_dir, "tydiqa-goldp-dev-" + lang + ".json")
                    scores = compute_scores(input_file, gold_file)
                    model_name = model
                    if "{" in model and "}" in model:
                        model_name = model.split("{")[1].split("}")[0]
                    if output_list:
                        with open(output_file, "a") as f:
                            for i in range(len(scores["f1_list"])):
                                f.write("{},{},{},{},{},{},{},{}\n".format(model_name, lang, decoding_strat, num_examples, scores["exact_match_list"][i], scores["f1_list"][i], scores["cfm_list"][i], scores["cfm_match_list"][i]))
                    else:
                        with open(output_file, "a") as f:
                            f.write("{},{},{},{},{},{},{},{}\n".format(model_name, lang, decoding_strat, num_examples, scores["exact_match"], scores["f1"], scores["cfm_score"], scores["cfm_match"]))

def main():
    args = parse_args()
    compute_scores_dir(args.input_dir, args.output_file, args.gold_dir, args.output_list)

if __name__ == "__main__":
    main()

# Example usage:

# GOLDP
# python postprocessing/tydiqa.py --input_dir ./outputs/tydiqa-goldp --output_file ./results/4_tydiqa/goldp.csv --gold_dir ./data/tydiqa/tydiqa-goldp-v1.1-dev/ --output_list

# CLOSED BOOK
# python postprocessing/tydiqa.py --input_dir ./outputs/tydiqa-closedbook --output_file ./results/4_tydiqa/closedbook.csv --gold_dir ./data/tydiqa/tydiqa-goldp-v1.1-dev/