import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

import argparse
import math
import random
import numpy as np
import json
import pandas as pd

from tqdm import tqdm

PROMPT_TEMPLATE = "<s>[INST] {question} </s> [/INST] {response}</s>"
DEFAULT_REWARD_MODEL = "berkeley-nest/Starling-RM-7B-alpha"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward_batch_size",
        type=int, 
        default=4,
        help="The batch size to use for reward model inference"
    )
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
        "--reward_model",
        type=str,
        default=DEFAULT_REWARD_MODEL,
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
    return parser.parse_args()

## Define the reward model function class

class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        device = "cuda"
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Using device:", device)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.device = device

    def get_device(self):
        return self.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores

## Load the model and tokenizer
def load_model(model_name=DEFAULT_REWARD_MODEL):
    reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
    reward_tokenizer = reward_model.tokenizer
    reward_tokenizer.truncation_side = "left"

    directory = snapshot_download(model_name)
    for fpath in os.listdir(directory):
        if fpath.endswith(".pt") or fpath.endswith("model.bin"):
            checkpoint = os.path.join(directory, fpath)
            break
    
    reward_model.load_state_dict(torch.load(checkpoint), strict=False)
    reward_model.eval().requires_grad_(False)
    reward_model.to(reward_model.get_device())

    return reward_model, reward_tokenizer

def get_rewards(samples, reward_batch_size, reward_model, reward_tokenizer):
    """samples: List[str]"""
    input_ids = []
    attention_masks = []
    encodings_dict = reward_tokenizer(
        samples,
        truncation=True,
        max_length=2048,
        padding="max_length",
        return_tensors="pt",
    ).to(reward_model.get_device())
    input_ids = encodings_dict["input_ids"]
    attention_masks = encodings_dict["attention_mask"]
    mbs = reward_batch_size
    out = []
    for i in tqdm(range(math.ceil(len(samples) / mbs)), leave=False):
        rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
        out.extend(rewards)
    return torch.hstack(out)

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

def generate_country_prompts(countries, prompt_template, add_article=True):
    prompts = []
    for country in countries:
        if add_article and should_add_article(country):
            if "[/INST] {country}" in prompt_template:
                prompts.append(prompt_template.format(country="The " + country))
            else:
                prompts.append(prompt_template.format(country="the " + country))
        else:
            prompts.append(prompt_template.format(country=country))
    return prompts

def main(args):
    reward_model, reward_tokenizer = load_model(args.reward_model)

    countries = get_countries()
    results = {"Question":[], "Full Prompt":[], "reward_model": [], "Search Query":[], "Sentiment":[], "Response Format":[], "Category":[], "seed": [], "add_article": []}
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

        prompt = PROMPT_TEMPLATE.format(question=question, response=response_format)
        prompts = generate_country_prompts(countries, prompt, args.add_article)

        rewards = get_rewards(prompts, args.reward_batch_size, reward_model, reward_tokenizer).tolist()

        results["Question"].append(question)
        results["Full Prompt"].append(prompt)
        results["reward_model"].append(args.reward_model)
        results["Search Query"].append(query)
        results["Sentiment"].append(sentiment)
        results["Response Format"].append(response_format)
        results["Category"].append(category)
        results["seed"].append(args.seed)
        results["add_article"].append(1 if args.add_article else 0)

        for j, country in enumerate(countries):
            results[country].append(rewards[j])

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