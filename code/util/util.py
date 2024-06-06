from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import os
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

####################################################################################
#                         Run Tests on Different Parts of ICE
####################################################################################

DOC_MAP = {
    'W1A': 'Non-Professional Writing',
    'W1B': 'Correspondence',
    'W2A': 'Academic Writing',
    'W2B': 'Non-Academic Writing',
    'W2C': 'Reportage',
    'W2D': 'Instructional Writing',
    'W2E': 'Persuasive Writing',
    'W2F': 'Creative Writing',
}

NIGERIA_MAP = {
    'AHUM': 'Academic Writing',
    'ANSC': 'Academic Writing',
    'ASSC': 'Academic Writing',
    'ATEC': 'Academic Writing',
    'ADM': 'Instructional Writing',
    'BL': 'Correspondence',
    'ED': 'Reportage',
    'EX': 'Non-Professional Writing',
    'NOV': 'Creative Writing',
    'PHUM': 'Non-Academic Writing',
    'PNSC': 'Non-Academic Writing',
    'PSSC': 'Non-Academic Writing',
    'PTEC': 'Non-Academic Writing',
    'PR': 'Reportage',
    'SKHO': 'Instructional Writing',
    'SL': 'Correspondence',
    'ESS': 'Non-Professional Writing'
}

def run_test_on_ice(ice_path:str, test:Any, test_name:str="test") -> pd.DataFrame:
    output = defaultdict(list)
    for dialect in os.listdir(ice_path):
        if(not dialect == ".DS_Store" and os.path.isdir(os.path.join(ice_path,dialect))):
            print("\n Running " + test_name + " for", dialect)
            for file in tqdm(os.listdir(os.path.join(ice_path,dialect))):
                if file == ".DS_Store" or file == ".DS_Store-0.txt":
                    continue
                with open(os.path.join(ice_path, dialect, file), "r") as f:
                    text = "".join(f.readlines())
                    if text == "":
                        continue
                    result = test(text)
                    output['title'].append(file)
                    output['dialect'].append(dialect)
                    output['length'].append(len(text))
                    output['lines'].append(text.count('\n'))
                    if (dialect == 'Nigeria'):
                        output['category'].append(NIGERIA_MAP[file.split('_')[0].upper()])
                    else:
                        output['category'].append(DOC_MAP[file[:3].upper()])
                    for key, value in result.items():
                        output[key].append(value)
    
    return pd.DataFrame(output)

####################################################################################
#                        Run Tests on Different Parts of MD3
####################################################################################

def get_md3_categories(md3_orig_path:str, dialect:str) -> dict:
    file_path = os.path.join(md3_orig_path, "prompts_" + dialect + ".tsv")
    df = pd.read_csv(file_path, delimiter="\t")
    mapping_dict = dict(zip(df["clip_identifier"], df["correct_word/image"]))
    return mapping_dict

def run_test_on_md3(md3_clean_path:str, md3_orig_path:str, test:Any, test_name:str="test") -> pd.DataFrame:
    output = defaultdict(list)
    for dialect in os.listdir(md3_clean_path):
        if(not dialect == ".DS_Store" and os.path.isdir(os.path.join(md3_clean_path,dialect))):
            print("\n Running " + test_name + " for", dialect)
            category_map = get_md3_categories(md3_orig_path, dialect)
            for file in tqdm(os.listdir(os.path.join(md3_clean_path,dialect))):
                if file == ".DS_Store" or file == ".DS_Store-0.txt":
                    continue
                with open(os.path.join(md3_clean_path, dialect, file), "r") as f:
                    text = "".join(f.readlines())
                    if text == "":
                        continue
                    result = test(text)
                    output['title'].append(file)
                    output['length'].append(len(text))
                    output['lines'].append(text.count('\n'))
                    output['speaker0'].append(text.count('Speaker0:'))
                    output['speaker1'].append(text.count('Speaker1:'))
                    output['dialect'].append(dialect)
                    output['category'].append(category_map[file[:-4]])
                    for key, value in result.items():
                        output[key].append(value)
    
    return pd.DataFrame(output)

####################################################################################
#                           Load Models and Tokenizers
####################################################################################

def load_models(model_id:str, load_8bit:bool):
    device = "cuda"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = None
    try:
        if(device == torch.device("cpu")):
            print("Loading model on CPU, 8-bit quantization will be disabled")
            model = AutoModelForCausalLM.from_pretrained(model_id)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=load_8bit)
        # if (not load_8bit):
        #     model = model.to(device)
    except Exception as e:
        if (device == torch.device("cpu")):
            print("Loading model on CPU, 8-bit quantization will be disabled")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=load_8bit)
        # if (not load_8bit):
        #     model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer