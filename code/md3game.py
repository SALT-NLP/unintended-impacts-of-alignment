import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import pandas as pd
import random
from tqdm import tqdm

model_prompt = """I am going to show you the transcript of a game two people are playing called Taboo.  The goal of the game is to guess the secret word without saying any of the distractor words.  Given the transcript, your goal is to guess the secret word.

Use the following format:
Transcript: The transcript between the two players.
Secret Word: The secret word that the guesser is trying to guess.

---

Transcript: "{transcript}"

Secret Word:"""

model_prompt_with_answers = """I am going to show you the transcript of a game two people are playing called Taboo.  The goal of the game is to guess the secret word without saying any of the distractor words.  Given the transcript and a set of possible answers, your goal is to guess the secret word.

Use the following format:
Transcript: The transcript between the two players.
Possible Answers: The possible answers that the guesser can choose from.
Secret Word: The secret word that the guesser is trying to guess.

---

Transcript: "{transcript}"

Possible Answers: {answers}

Secret Word:"""

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
        "--md3_input_dir",
        type=str,
        default="./",
        help="The path to the md3 corpus",
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
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="no_answers",
        help="The mode to use for the prompt.  Either 'answers' or 'no_answers'.",
        choices=["answers", "no_answers"]
    )
    parser.add_argument(
        "--use_chat_template",
        action='store_true',
        default=False,
        help="Whether or not to use the chat template for the prompt",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="The maximum number of new tokens to generate"
    )

    args = parser.parse_args()

    return args

skip_set = {
    "3unebi8s_r03_s0_p004", # This one they just say "ready?"
    "3unebi8s_r03_s0_p005", # This one is blank
    "3unebi8s_r03_s0_p007", # This one is blank
    "a73b4r3y_r03_s1_p000", # This one uses time sensitive knowledge (september is in 30 days)
    "bkp15ow3_r03_s1_p007", # This one they say "same word as before"
    "20y5iewb_r02_s0_p011", # Correct word does not match transcript
    "6c2o0pur_r03_s1_p000", # No discussion related to the game
    "6c2o0pur_r03_s1_p001", # Did not actually complete the game this turn
    "6c2o0pur_r03_s1_p002", # Correct word does not match transcript
    "6c2o0pur_r03_s1_p003", # Correct word does not match transcript
    "6c2o0pur_r03_s1_p005", # Correct word does not match transcript
}

alternative_words = {
    # en_in
    "0iv2hqsg_r03_s1_p003": "Throwing caramel",
    "0iv2hqsg_r03_s1_p006": None,
    "1w0fbf7w_r02_s0_p004" : "fellow",
    "2dj639rm_r02_s0_p004": "Rihana",
    "2dj639rm_r02_s0_p005": "Java script",
    "2j1kw67f_r02_s1_p001": None,
    "2j1kw67f_r02_s1_p003": None,
    "2j1kw67f_r03_s1_p004": "Billy holiday",
    "2j1kw67f_r03_s1_p018": "will search",
    "2snk9xzj_r03_s0_p001": "ah leb ah",
    "4v5v7c44_r03_s0_p002": "grate wall of China",
    "4v5v7c44_r03_s0_p004": None,
    "5p6nnuq5_r02_s1_p000": "hang man",
    "5p6nnuq5_r02_s1_p002": None,
    "5p6nnuq5_r03_s1_p000": "Delhi holiday",
    "5wjo0hxg_r02_s0_p002": None,
    "5wjo0hxg_r02_s0_p004": None,
    "7q20i19r_r02_s1_p002": "cow boy",
    "8fcwqowm_r03_s0_p004": "Calvin",
    "8kyo8kbk_r03_s0_p001": "Raymon birds",
    "8v89jv2b_r02_s1_p000": None,
    "8v89jv2b_r03_s1_p000": "stage coach",
    "8v89jv2b_r03_s1_p002": "aluminum",
    "99oa08is_r02_s1_p000": "Cooling",
    "aeu8hb63_r02_s1_p000": "C E N S O R S H I P",
    "aeu8hb63_r03_s1_p003": None,
    "az2d5v9v_r02_s1_p001": "World Disney World",
    "az2d5v9v_r03_s1_p000": "solder",
    "az2d5v9v_r03_s1_p003": "Tiffs",
    "cat80g8y_r02_s1_p001": None,
    "d6mm3ola_r03_s1_p006": "Ascel",
    "dwmqwtdg_r02_s0_p001": "goes't",
    "dwmqwtdg_r02_s0_p011": None,
    "dwmqwtdg_r03_s0_p013": "C programing language",
    "dwmqwtdg_r03_s0_p016": "first aid kit",
    "dwmqwtdg_r03_s0_p017": "favorite hero",
    "e8rznyof_r02_s0_p002": "Mahmood Ali",
    "eazsmxi9_r02_s0_p002": None,
    "eazsmxi9_r02_s0_p003": None,
    "eazsmxi9_r02_s0_p004": "Hero",
    "euwjjcge_r02_s0_p001": None,
    "euwjjcge_r03_s0_p001": "ECO",
    "euwjjcge_r03_s0_p005": "first aid kit",
    "ex5khl6g_r02_s1_p001": "Whitehouse",
    "f3sptm6u_r02_s1_p012": None,
    "5wjo0hxg_r02_s0_p003": None,

    # en_us
    "03hxmcfy_r02_s1_p002": "Egypt",
    "03hxmcfy_r03_s1_p003": None,
    "03hxmcfy_r03_s1_p007": "Meatballs",
    "0oenipov_r02_s0_p003": None,
    "0oenipov_r02_s0_p004" : None,
    "0oenipov_r02_s0_p006" : None,
    "0oenipov_r03_s0_p000" : "Churchill",
    "0oenipov_r03_s0_p001" : "Shaq O'Neal",
    "17h8rpo0_r02_s0_p001" : "Russian",
    "17h8rpo0_r02_s0_p011" : "Anchovies",
    "17h8rpo0_r02_s0_p013" : None,
    "17h8rpo0_r03_s1_p007" : None,
    "17h8rpo0_r03_s1_p009" : "Disney",
    "17h8rpo0_r03_s1_p010" : None,
    "37z6z85p_r02_s0_p001" : None,
    "3unebi8s_r02_s0_p001" : None,
    "3unebi8s_r02_s0_p002" : None,
    "3unebi8s_r02_s0_p003" : None,
    "48gr65ds_r03_s1_p001" : None,
    "4g5ds93k_r03_s0_p006" : "Notre Dame",
    "4ye4o14q_r03_s1_p004" : None,
    "5e2sjzlp_r02_s1_p002" : None,
    "5e2sjzlp_r03_s1_p002" : "Lefthanded",
    "5eu3vyzv_r03_s1_p004" : None,
    "5gbt496n_r03_s0_p004" : "hairbrush",
    "5gbt496n_r03_s0_p005" : "fishing",
    "5jf6jk9p_r02_s0_p012" : None,
    "5jf6jk9p_r02_s0_p016" : "2Pac",
    "5jf6jk9p_r03_s0_p000" : "Ghengis Khan",
    "5jf6jk9p_r03_s0_p002" : "JFK Airport",
    "60wbpmd7_r03_s0_p011" : "alternative",
    "6lbpt3ef_r03_s0_p002" : "Madonna",
    "74rx784t_r02_s0_p008" : "e commerce",
    "81l5v8nw_r03_s1_p001" : "Kim",
    "81l5v8nw_r03_s1_p004" : "Microsoft",
    "81l5v8nw_r03_s1_p005" : "body mass, index",
    "8gmkwlei_r03_s1_p000" : "Diana Princess of Wales",
    "8kfg4tl5_r02_s1_p001" : "Rats",
    "8kfg4tl5_r03_s1_p000" : "like lip",
    "8vdxh7ux_r03_s0_p003" : "Genetic",
    "a73b4r3y_r02_s1_p004" : "Computer program",
    "a73b4r3y_r02_s1_p010" : "Anchovies",
    "a73b4r3y_r03_s1_p000" : None,
    "a73b4r3y_r03_s1_p009" : "Carpet or",
    "aprbao97_r02_s0_p001" : "Little Red Ridding Hood",
    "aprbao97_r02_s0_p004" : None,
    "avzhhqzg_r03_s1_p008" : "Windows",
    "bkp15ow3_r03_s1_p010" : "solar panel",
    "bkp15ow3_r03_s1_p012" : None,
    "bkp15ow3_r03_s1_p015" : "Spiderman",
    "blsu4hu5_r03_s1_p012" : "diving board",
    "cwaj60h3_r03_s0_p002" : "meatballs",
    "dam6dz98_r02_s0_p000" : None,
    "dam6dz98_r02_s0_p001" : None,
    "dam6dz98_r02_s0_p002" : None,
    "dam6dz98_r02_s0_p003" : None,
    "dam6dz98_r02_s0_p004" : None,
    "dam6dz98_r02_s0_p005" : None,
    "dam6dz98_r02_s0_p006" : None,
    "dam6dz98_r02_s0_p007" : None,
    "eepaxoxo_r02_s0_p003" : "Fishing",
    "eepaxoxo_r02_s0_p008" : "micro",
    "eepaxoxo_r02_s0_p009" : None,
    "eepaxoxo_r03_s0_p006" : "Madonna",
    "eepaxoxo_r03_s0_p010" : "Hypnotic",
    "eepaxoxo_r03_s0_p011" : "jiu-jitsu",
    "eki0yxp0_r02_s1_p003" : None,
    "eugxlgs7_r02_s1_p002" : "homophobic",
    "eugxlgs7_r02_s1_p006" : None,
    "eugxlgs7_r03_s1_p012" : None,
    "eugxlgs7_r03_s1_p013" : None,
    "evgjkid4_r02_s1_p004" : "bike",
    "evgjkid4_r03_s0_p014" : "Mutant",

    # en_ng
    "0iozllt3_r03_s1_p000" : None,
    "0iozllt3_r03_s1_p001" : None,
    "0iozllt3_r03_s1_p002" : None,
    "0iozllt3_r03_s1_p003" : None,
    "0iozllt3_r03_s1_p004" : None,
    "0k9nzctj_r02_s0_p004" : None,
    "0k9nzctj_r02_s0_p008" : "Christiano Ronaldo",
    "0k9nzctj_r02_s0_p015" : None,
    "0s6qjbse_r03_s1_p000" : None,
    "0ws3dc75_r02_s1_p003" : "toll truck",
    "0ws3dc75_r03_s1_p002" : None,
    "0znoh0nk_r02_s1_p014" : "Beyonce",
    "0znoh0nk_r03_s1_p001" : "screw driver",
    "1qy6rzn9_r02_s1_p007" : "Kayne West",
    "20y5iewb_r02_s0_p008" : None,
    "20y5iewb_r03_s0_p007" : None,
    "23t73nau_r02_s0_p004" : None,
    "5ilw47ag_r02_s1_p001" : "Princess Diana",
    "5vzitx0p_r02_s0_p001" : None,
    "6qobqnlu_r02_s0_p003" : "Spiderman",
    "6qobqnlu_r03_s0_p001" : "Infertile",
    "7gxzb7cg_r02_s0_p005" : "Programming language",
    "7vdbgsdm_r03_s1_p000" : "Prince Charles of Wales",
    "7vdbgsdm_r03_s1_p006" : None,
    "7vdbgsdm_r03_s1_p007" : None,
    "7vdbgsdm_r03_s1_p009" : None,
    "8pfjjlr3_r03_s1_p000" : "Brad Piit",
    "ajwnz2l9_r03_s1_p002" : "Microsoft",
    "ajwnz2l9_r03_s1_p004" : "joy stick",
    "ajwnz2l9_r03_s1_p008" : "J.F Kennedy",
    "augohsp9_r02_s0_p000" : "Queen Elizabeth the second",
    "augohsp9_r03_s0_p005" : "Jenifer Lawrence",
    "av10dhru_r02_s1_p012" : "Cow boys",
    "av10dhru_r02_s1_p016" : None,
    "b0chmq0q_r02_s0_p009" : None,
    "b836rybj_r02_s0_p006" : "PC",
    "b836rybj_r03_s0_p004" : "Accounts",
    "cbonciz9_r02_s1_p000" : "Marvel",
    "cbonciz9_r02_s1_p006" : "Zlatan Ibrahimovic",
    "disozw4t_r02_s1_p000" : "Fungi",
    "dnt3gj14_r03_s0_p005" : "Blackhole",
    "e1dcr8hf_r02_s1_p001" : "Coby Bryant",
    "e1dcr8hf_r02_s1_p002" : "Android",
    "e1dcr8hf_r03_s1_p001" : "Olympics games",
}

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

def generate_answer(prompt, tokenizer, model, device="cuda", max_new_tokens=10):
    prompt_input = tokenizer(prompt, return_tensors='pt').to(device)
    prompt_tokens = prompt_input["input_ids"]
    attn_mask = prompt_input["attention_mask"]
    prompt_length = len(prompt_tokens[0])

    output = model.generate(input_ids=prompt_tokens, attention_mask=attn_mask, do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=max_new_tokens)

    output = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

    return output.strip()

def load_data(data_path: str, dataset: str):
    tsv_file = os.path.join(data_path, "md3", "prompts_" + dataset + ".tsv")
    data = pd.read_csv(tsv_file, sep='\t')

    dataset_output = []

    # Filter the data to just game_type "word" and prompt_status "win"
    filtered_data = data[data["game_type"] == "word"]
    filtered_data = filtered_data[filtered_data["prompt_status"] == "win"]

    for row in filtered_data.iterrows():
        clip_identifier = row[1]["clip_identifier"]
        if (clip_identifier in skip_set):
            continue
        correct_word = str(row[1]["correct_word/image"])
        distractors = row[1]["distractors"]
        transcript_file = os.path.join(data_path, "md3_clean", dataset, clip_identifier + ".txt")
        if (os.path.exists(transcript_file)):
            with open(transcript_file, 'r') as f:
                transcript = f.readlines()
                ans_line = -1
                word = correct_word
                if (clip_identifier in alternative_words.keys()):
                    if (alternative_words[clip_identifier] is not None):
                        word = alternative_words[clip_identifier]
                    else:
                        ans_line = len(transcript)
                        word = None
                if word is not None:
                    word = word.strip().lower()

                    for line_num in range(len(transcript)):
                        if (transcript[line_num].strip().lower().find(word) != -1):
                            ans_line = line_num
                            break
                    if (ans_line == -1):
                        print("Correct word " + correct_word + " not found in transcript: " + transcript_file)
                        print(clip_identifier)
                        continue

                transcript_cut = ''.join(transcript[:ans_line]).strip()

                row = {"transcript": transcript_cut, "correct_word": correct_word, "distractors": eval(distractors)}
                dataset_output.append(row)

        else:
            print("Transcript file does not exist: " + transcript_file)
            continue

    return dataset_output

    

def compute_md3_wordgame_scores(model_id:str, data_path:str, output_file:str, load_8bit:bool, prompt_mode:str, seed:int, use_chat_template:bool, max_new_tokens:int=10):
    device = "cuda"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=load_8bit)
        if (not load_8bit):
            model = model.to(device)
    except Exception as e:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', load_in_8bit=load_8bit)
        if (not load_8bit):
            model = model.to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the data
    data_in = load_data(data_path, "en_in")
    data_ng = load_data(data_path, "en_ng")
    data_us = load_data(data_path, "en_us")

    outputs =  {"model": [],
                "load_8bit": [],
                "prompt_mode": [],
                "seed": [],
                "dialect": [],
                "correct_answer_prob": [], 
                "distractor_probs": [],
                "generated_answer": [],
                "chose_correct_answer": [],
                "chose_correct_answer_num": [],
                "generated_correct_answer": [],
                "generated_correct_answer_num": [],
                "transcript":[], 
                "correct_answer": [], 
                "distractors": [],
                "using_chat_template": []}
    
    for dialect, dataset in [("en_in", data_in), ("en_ng", data_ng), ("en_us", data_us)]:
        for row in tqdm(dataset):
            transcript = row["transcript"]
            answers = row["distractors"]
            correct_word = row["correct_word"]

            correct_idx = random.randint(0, len(answers) - 1)
            all_answers = answers[:correct_idx] + [correct_word] + answers[correct_idx:]

            prompt = "ERROR ERROR ERROR ERROR ERROR"
            if (prompt_mode == "answers"):
                prompt = model_prompt_with_answers.format(transcript=transcript, answers=all_answers)
            elif (prompt_mode == "no_answers"):
                prompt = model_prompt.format(transcript=transcript)

            if (use_chat_template):
                chat = [
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            losses = compute_negative_log_likelihoods(prompt, all_answers, tokenizer, model, device=device)
            probs = torch.exp(losses * -1)
            probs = probs / torch.sum(probs)

            max_choice = torch.argmax(probs).item()

            generation = generate_answer(prompt, tokenizer, model, device=device, max_new_tokens=max_new_tokens)

            outputs["model"].append(model_id)
            outputs["load_8bit"].append(load_8bit)
            outputs["prompt_mode"].append(prompt_mode)
            outputs["seed"].append(seed)
            outputs["dialect"].append(dialect)
            outputs["correct_answer_prob"].append(probs[correct_idx].item())
            outputs["distractor_probs"].append(probs[:correct_idx].tolist() + probs[correct_idx + 1:].tolist())
            outputs["generated_answer"].append(generation)
            outputs["chose_correct_answer"].append(max_choice == correct_idx)
            outputs["chose_correct_answer_num"].append(1 if max_choice == correct_idx else 0)
            outputs["transcript"].append(transcript)
            outputs["correct_answer"].append(correct_word)
            outputs["distractors"].append(answers)

            generated_correct_answer = False
            if (correct_word.lower() in generation.lower()):
                check_other_words = False
                for word in answers:
                    if (word.lower() in generation.lower()):
                        check_other_words = True
                        break
                if (not check_other_words):
                    generated_correct_answer = True

            outputs["generated_correct_answer"].append(generated_correct_answer)
            outputs["generated_correct_answer_num"].append(1 if generated_correct_answer else 0)
            outputs["using_chat_template"].append(use_chat_template)

    # Create the output directory if it doesn't exist
    if os.path.dirname(output_file) != "" and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # Write the results to the output directory
    df = pd.DataFrame(outputs)
    df.to_csv(output_file, index=False)

def main():
    args = parse_args()
        
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    compute_md3_wordgame_scores(args.model_id, args.md3_input_dir, args.output_file, args.load_8bit, args.prompt_mode, args.seed, args.use_chat_template, args.max_new_tokens)

if __name__ == "__main__":
    main()

# Example Usage:
# python code/md3game.py --model_id gpt2 --output_file output.csv --md3_input_dir ./data/md3/ --prompt_mode no_answers --seed 221