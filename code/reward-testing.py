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

DEFAULT_PROMPT = "<s>[INST] Where are you from? </s> [/INST] I am from {country}.</s>"
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
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="The prompt to use",
    )
    parser.add_argument(
        "--add_article",
        action="store_true",
        default=True,
        help="Whether to add an article to the country name in the prompt when appropriate",
    )
    parser.add_argument(
        "--prompt_id",
        type=str,
        default="default",
        help="The identifier to use for the prompt",
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

def get_rewards(prompts, reward_batch_size=4, reward_model_name=DEFAULT_REWARD_MODEL):

    reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
    reward_tokenizer = reward_model.tokenizer
    reward_tokenizer.truncation_side = "left"

    directory = snapshot_download(reward_model_name)
    for fpath in os.listdir(directory):
        if fpath.endswith(".pt") or fpath.endswith("model.bin"):
            checkpoint = os.path.join(directory, fpath)
            break
    
    reward_model.load_state_dict(torch.load(checkpoint), strict=False)
    reward_model.eval().requires_grad_(False)
    reward_model.to(reward_model.get_device())


    ## Define the reward function

    def get_reward(samples):
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
        for i in tqdm(range(math.ceil(len(samples) / mbs))):
            rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
            out.extend(rewards)
        return torch.hstack(out)

    ## Inference over prompts with llama2 chat template

    return get_reward(prompts)

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
            prompts.append(prompt_template.format(country="the " + country))
        else:
            prompts.append(prompt_template.format(country=country))
    return prompts

def main(args):
    countries = get_countries()
    prompts = generate_country_prompts(countries, args.prompt, args.add_article)

    rewards = get_rewards(prompts, args.reward_batch_size).tolist()

    results = {}
    results["country"] = countries
    results["prompt"] = prompts
    results["reward"] = rewards
    results["seed"] = [args.seed] * len(prompts)
    results["reward_model"] = [args.reward_model] * len(prompts)
    results["prompt_id"] = [args.prompt_id] * len(prompts)
    results["add_article"] = [1 if args.add_article else 0] * len(prompts)

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
# python code/reward-testing.py --reward_batch_size 4 --seed 221 --output_file output.csv
# python code/reward-testing.py --reward_batch_size 4 --seed 221 --output_file output.csv --prompt "<s>[INST] What are good countries to raise a family? </s> [/INST] One good country to raise a family in is {country}.</s>" --add_article --prompt_id "family"