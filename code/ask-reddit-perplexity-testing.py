import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download

import argparse
import math
import random
import numpy as np
import json
import pandas as pd

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=221,
        help="The random seed to use"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.csv",
        help="The file to output results to",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The reward model to use",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="../data/askreddit_countries/AskRedditCountries_final.csv",
        help="The path to the input file",
    )
    parser.add_argument(
        "--add_article",
        action="store_true",
        default=True,
        help="Whether to add an article to the country name in the prompt when appropriate",
    )
    parser.add_argument(
        "--reformat_response",
        action="store_true",
        default=False,
        help="Whether to reformat the response to be more natural",
    )
    parser.add_argument(
        "--load_8bit",
        action='store_true',
        default=False,
        help="Whether or not to load the model with 8bit quantization",
    )
    parser.add_argument(
        "--use_chat_template",
        action='store_true',
        default=False,
        help="Whether or not to use the chat template for the prompt",
    )
    return parser.parse_args()

## Load the model and tokenizer
def load_model(model_id, load_8bit=False, device=torch.device('cuda')):
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=load_8bit)
        if (not load_8bit):
            model = model.to(device)
    except Exception as e:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', load_in_8bit=load_8bit)
            if (not load_8bit):
                model = model.to(device)
        except Exception as e:
            print("Unable to load in 8bit, trying with full precision")
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.eval()

    return model, tokenizer

def compute_negative_log_likelihoods(prompt, answers, tokenizer, model, device="cuda"):
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = len(prompt_tokens[0])

    losses = []

    for answer in answers:
        inputs = tokenizer.encode(prompt + answer, return_tensors='pt').to(device)
        labels = inputs.clone()
        labels[:, :prompt_length] = -100

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs.loss

        losses.append(loss.item())

    return torch.Tensor(losses)

def get_countries():
    countries = []
    with open("../data/countries/countries.geojson") as f:
        json_data = json.load(f)
        for feature in json_data["features"]:
            countries.append(feature["properties"]["ADMIN"])
    
    return countries

def should_add_article(country):
    # List of keywords for which "The" should typically be prepended
    keywords = ['Islands', 'Island', 'States', 'Republic', 'United']

    # Special hardcoded cases
    special_cases = [
        'Bahamas',
        'Gambia',
        'Netherlands',
        'Ivory Coast',
        'Philippines',
        'Maldives',
        'Seychelles',
        'Russian Federation',
        'Vatican',
    ]

    # Check if the country name matches any special cases
    if country in special_cases:
        return True

    # Check if the country name contains any of the keywords
    for keyword in keywords:
        if keyword in country:
            return True

    # If no match, return the original name
    return False

# Generate the prompts for each country stopping at the country name for perplexity measurement
def generate_country_prompts(countries, question, response_format, tokenizer, add_article=True, use_chat_template=False):
    prompt = question
    if use_chat_template:
        chat = [
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    response_prefix = ""
    if not response_format.startswith("{country}"):
        response_prefix = response_format.split("{country}")[0]

    prompt = prompt + " " + response_prefix
    prompt = prompt.strip()
    answers = []
    for country in countries:
        if add_article and should_add_article(country):
            if response_format.startswith("{country}"):
                answers.append(" The " + country)
            else:
                answers.append(" the " + country)
        else:
            answers.append(" " + country)
    return prompt, answers

def main(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print("Using device:", device)

    model, tokenizer = load_model(args.model_id, args.load_8bit, device)

    countries = get_countries()
    results = {"Question":[], "Full Prompt":[], "model": [], "Search Query":[], "Sentiment":[], "Response Format":[], "Category":[], "seed": [], "add_article": [], "use_chat_template": []}
    for country in countries:
        results[country] = []

    skip = 0
    if args.output_file != "" and os.path.exists(args.output_file):
        hot_start = pd.read_csv(args.output_file)
        for col in hot_start.columns:
            results[col] = hot_start[col].tolist()
        skip = len(hot_start)

    input_df = pd.read_csv(args.input_file)
    for i in tqdm(range(len(input_df)), desc="All Prompts"):
        if i < skip:
            continue

        query = input_df['Search Query'][i]
        sentiment = input_df['Sentiment'][i]
        question = input_df['Questions'][i]
        response_format = input_df['Response Format'][i]
        category = input_df['Category'][i]

        if not args.reformat_response:
            response_format = "{country}."
        
        prompt, answers = generate_country_prompts(countries, question, response_format, tokenizer, args.add_article, use_chat_template=args.use_chat_template)

        nlls = compute_negative_log_likelihoods(prompt, answers, tokenizer, model, device=device)

        results["Question"].append(question)
        results["Full Prompt"].append(prompt)
        results["model"].append(args.model_id)
        results["Search Query"].append(query)
        results["Sentiment"].append(sentiment)
        results["Response Format"].append(response_format)
        results["Category"].append(category)
        results["seed"].append(args.seed)
        results["add_article"].append(1 if args.add_article else 0)
        results["use_chat_template"].append(1 if args.use_chat_template else 0)

        for j, country in enumerate(countries):
            results[country].append(nlls[j].item())

        if i % 10 == 0:
            if os.path.dirname(args.output_file) != "" and not os.path.exists(os.path.dirname(args.output_file)):
                os.makedirs(os.path.dirname(args.output_file))

            df = pd.DataFrame(results)
            df.to_csv(args.output_file, index=False)

    if os.path.dirname(args.output_file) != "" and not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)

    

# Example usage:
# python code/ask-reddit-reward-testing.py --reward_batch_size 4 --seed 221 --output_file output.csv
# python code/ask-reddit-reward-testing.py --reward_batch_size 4 --seed 221 --output_file output.csv --add_article
    
# python ask-reddit-reward-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-reward-testing/gpt2-reformatted.csv --add_article --model_id gpt2 --reformat_response