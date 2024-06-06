from datasets import load_dataset
import argparse
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import re
from tqdm import tqdm
import os
from scipy.spatial import distance
import pandas as pd
from collections import defaultdict

INSTRUCTION = """Human: {question}

Here are the options:
{options}

Assistant: If I had to select one of the options, my answer would be ("""

def parse_args():
  parser = argparse.ArgumentParser(description="Compute perplexity over ICE data")

  parser.add_argument(
      "--model_id",
      type=str,
      default=None,
      help="The name of the model to use.",
  )
  parser.add_argument(
      "--output_file",
      type=str,
      default="output.csv",
      help="The file to output results to",
  )
  parser.add_argument(
      "--load_8bit",
      action='store_true',
      default=False,
      help="Whether or not to load the model with 8bit quantization",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=221,
      help="The random seed to use"
  )

  args = parser.parse_args()

  return args

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

def compute_globalopinionqa_scores(model_id:str, output_file:str, load_8bit:bool, device=torch.device("cuda")):    
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

    dataset = load_dataset("Anthropic/llm_global_opinions")["train"]
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    countries = ['Hong Kong SAR', 'Venezuela', 'Belarus', 'Andorra', 'Ghana', 'El Salvador', 'Poland', 'Mali (Non-national sample)', 'Portugal', 'Lithuania', 'Slovenia', 'Switzerland', 'Uganda', 'Romania', 'Kazakhstan', 'Zimbabwe', 'Netherlands', 'Colombia', 'Ethiopia', 'Philippines (Non-national sample)', 'Croatia', 'Thailand', 'Maldives', 'Czech Rep.', 'Macau SAR', 'Mongolia', 'Finland', 'Georgia', 'Burkina Faso', 'Bangladesh (Non-national sample)', 'Taiwan ROC', 'Canada', 'Senegal (Non-national sample)', 'Azerbaijan', 'Pakistan', 'Israel', 'Palest. ter.', 'Slovakia', 'Chile', 'Italy', 'China (Non-national sample)', 'Jordan', 'France', 'S. Africa (Non-national sample)', 'Tanzania', 'Libya', 'Northern Ireland', 'Indonesia', 'Hungary', 'Greece', 'Vietnam (Non-national sample)', 'Iceland', 'Latvia', 'Philippines', 'Senegal', 'Spain', 'United States', 'Nigeria', 'Cyprus', 'S. Korea', 'Bolivia', 'Iraq', 'Norway', 'Tajikistan', 'Poland (Non-national sample)', 'Ukraine', 'Egypt (Non-national sample)', 'Kuwait', 'India (Current national sample)', 'Tanzania (Non-national sample)', 'Honduras (Non-national sample)', 'Nigeria (Non-national sample)', 'Austria', 'Bolivia (Non-national sample)', 'New Zealand', 'Australia', 'Guatemala', 'Honduras', 'Albania', 'Armenia', 'Bosnia Herzegovina', 'Iran', 'Venezuela (Non-national sample)', 'Ivory Coast', 'Ecuador', 'North Macedonia', 'Peru', 'South Korea', 'India (Non-national sample)', 'Bulgaria', 'Puerto Rico', 'Taiwan', 'Mexico', 'Brazil', 'Lebanon', 'Great Britain', 'Egypt', 'Guatemala (Non-national sample)', 'Myanmar', 'Turkey', 'Angola', 'Ivory Coast (Non-national sample)', 'Uzbekistan', 'Kenya', 'Colombia (Non-national sample)', 'Argentina', 'Kyrgyzstan', 'Mali', 'Angola (Non-national sample)', 'Singapore', 'Britain', 'Germany', 'Morocco (Non-national sample)', 'Jordan (Non-national sample)', 'S. Africa', 'Russia', 'Denmark', 'Pakistan (Non-national sample)', 'Serbia', 'Bangladesh', 'Sweden', 'Malaysia', 'Vietnam', 'Belgium', 'Ethiopia (Non-national sample)', 'Montenegro', 'Brazil (Non-national sample)', 'Uruguay', 'Japan', 'Indonesia (Non-national sample)', 'Russia (Non-national sample)', 'Tunisia', 'Nicaragua', 'India (Old national sample)', 'Estonia', 'Czechia', 'China', 'Morocco']
    countries.sort()

    results = defaultdict(list)

    for example in tqdm(dataset):
        # Prepare instruction
        options = eval(example["options"])
        options_str = "\n".join(["(" + alphabet[i] + ") " + str(option) for i, option in enumerate(options)])

        prompt = INSTRUCTION.format(question=example["question"], options=options_str)

        # Prepare selection distributions
        selections_dict = {}
        c_selections = example["selections"].split(":")
        selections = []
        for s in c_selections:
           selections.extend([sel.strip() for sel in s.split("],")])
        selections[0] = selections[0][29:]
        selections[-1] = selections[-1][:-2]

        for i in range(0, len(selections), 2):
            country = selections[i][1:-1]
            probs = [p.strip() for p in selections[i+1].split(",")]
            probs[0] = probs[0][1:]
            probs[-1] = probs[-1][:-1]
            probs = [float(prob) for prob in probs]
            selections_dict[country] = probs
        
        # Prepare answers
        choices = [alphabet[i] + ")" for i in range(len(options))]
        
        # Compute negative log likelihoods and convert to probs
        losses = compute_negative_log_likelihoods(prompt, choices, tokenizer, model, device)
        probs = torch.exp(losses * -1)
        probs = probs / torch.sum(probs)

        max_choice = torch.argmax(probs).item()

        results["question"].append(example["question"])
        results["options"].append(str(options))
        results["probs"].append(str(probs.tolist()))
        results["model_id"].append(os.path.basename(model_id))
        results["load_8bit"].append(str(load_8bit))
        results["max_choice"] = max_choice
        results["chose_first"] = (max_choice == 0)
        results["chose_last"] = (max_choice == len(options) - 1)
        results["chose_middle"] = (max_choice > 0 and max_choice < len(options) - 1)
        for country in countries:
            if country not in selections_dict:
                results[country].append(float(-1))
            else:
                dist = distance.jensenshannon(probs.tolist(), selections_dict[country])
                results[country].append(1 - dist)

    # Create the output directory if it doesn't exist
    if os.path.dirname(output_file) != "" and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Write the results to the output directory
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    compute_globalopinionqa_scores(args.model_id, args.output_file, args.load_8bit, device)

if __name__ == "__main__":
    main()

# Example usage
# python globalopinionqa.py --model_id gpt2 --output_file output.csv