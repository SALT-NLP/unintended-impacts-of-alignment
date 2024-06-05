import argparse
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
import random

prompt = """Given the following passage, please answer the following question.  Use the following format:

Context: A passage containing the answer to the question.
Question: The question being asked.
Choices: The possible answers to the question.
Based on the choices the answer is: The correct answer to the question: A, B, C, or D.

---

Context: {context}
Question: {question}
Choices: {choices}
Based on the choices the answer is: """

def parse_args():
    parser = argparse.ArgumentParser(description="Compute scores for TyDi QA GoldP")

    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="The name of the model to use.",
    )
    parser.add_argument(    
        "--model_name",
        type=str,
        default=None,
        help="The name of the model to use.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output/",
        help="The directory to output results to",
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

def convert_iso_code_to_language_name(code):
    language_map = {
        "acm_Arab": "Mesopotamian Arabic",
        "afr_Latn": "Afrikaans",
        "als_Latn": "Tosk Albanian",
        "amh_Ethi": "Amharic",
        "apc_Arab": "North Levantine Arabic",
        "arb_Arab": "Modern Standard Arabic",
        "arb_Latn": "Modern Standard Arabic (Romanized)",
        "ars_Arab": "Najdi Arabic",
        "ary_arab": "Moroccan Arabic",
        "arz_Arab": "Egyptian Arabic",
        "asm_Beng": "Assamese",
        "azj_Latn": "North Azerbaijani",
        "bam_Latn": "Bambara",
        "ben_Beng": "Bengali",
        "ben_Latn": "Bengali (Romanized)",
        "bod_Tibt": "Standard Tibetan",
        "bul_Cyrl": "Bulgarian",
        "cat_Latn": "Catalan",
        "ceb_Latn": "Cebuano",
        "ces_Latn": "Czech",
        "ckb_Arab": "Central Kurdish",
        "dan_Latn": "Danish",
        "deu_Latn": "German",
        "ell_Grek": "Greek",
        "eng_Latn": "English",
        "est_Latn": "Estonian",
        "eus_Latn": "Basque",
        "fin_Latn": "Finnish",
        "fra_Latn": "French",
        "fuv_Latn": "Nigerian Fulfulde",
        "gaz_Latn": "West Central Oromo",
        "grn_Latn": "Guarani",
        "guj_Gujr": "Gujarati",
        "hat_Latn": "Haitian Creole",
        "hau_Latn": "Hausa",
        "heb_Hebr": "Hebrew",
        "hin_Deva": "Hindi",
        "hin_Latn": "Hindi (Romanized)",
        "hrv_Latn": "Croatian",
        "hun_Latn": "Hungarian",
        "hye_Armn": "Armenian",
        "ibo_Latn": "Igbo",
        "ilo_Latn": "Ilocano",
        "ind_Latn": "Indonesian",
        "isl_Latn": "Icelandic",
        "ita_Latn": "Italian",
        "jav_Latn": "Javanese",
        "jpn_Jpan": "Japanese",
        "kac_Latn": "Jingpho",
        "kan_Knda": "Kannada",
        "kat_Geor": "Georgian",
        "kaz_Cyrl": "Kazakh",
        "kea_Latn": "Kabuverdianu",
        "khk_Cyrl": "Halh Mongolian",
        "khm_Khmr": "Khmer",
        "kin_Latn": "Kinyarwanda",
        "kir_Cyrl": "Kyrgyz",
        "kor_Hang": "Korean",
        "lao_Laoo": "Lao",
        "lin_Latn": "Lingala",
        "lit_Latn": "Lithuanian",
        "lug_Latn": "Ganda",
        "luo_Latn": "Luo",
        "lvs_Latn": "Standard Latvian",
        "mal_Mlym": "Malayalam",
        "mar_Deva": "Marathi",
        "mkd_Cyrl": "Macedonian",
        "mlt_Latn": "Maltese",
        "mri_Latn": "Maori",
        "mya_Mymr": "Burmese",
        "nld_Latn": "Dutch",
        "nob_Latn": "Norwegian Bokmål",
        "npi_Deva": "Nepali",
        "npi_Latn": "Nepali (Romanized)",
        "nso_Latn": "Northern Sotho",
        "nya_Latn": "Nyanja",
        "ory_Orya": "Odia",
        "pan_Guru": "Eastern Panjabi",
        "pbt_Arab": "Southern Pashto",
        "pes_Arab": "Western Persian",
        "plt_Latn": "Plateau Malagasy",
        "pol_Latn": "Polish",
        "por_Latn": "Portuguese",
        "ron_Latn": "Romanian",
        "rus_Cyrl": "Russian",
        "shn_Mymr": "Shan",
        "sin_Latn": "Sinhala (Romanized)",
        "sin_Sinh": "Sinhala",
        "slk_Latn": "Slovak",
        "slv_Latn": "Slovenian",
        "sna_Latn": "Shona",
        "snd_Arab": "Sindhi",
        "som_Latn": "Somali",
        "sot_Latn": "Southern Sotho",
        "spa_Latn": "Spanish",
        "srp_Cyrl": "Serbian",
        "ssw_Latn": "Swati",
        "sun_Latn": "Sundanese",
        "swe_Latn": "Swedish",
        "swh_Latn": "Swahili",
        "tam_Taml": "Tamil",
        "tel_Telu": "Telugu",
        "tgk_Cyrl": "Tajik",
        "tgl_Latn": "Tagalog",
        "tha_Thai": "Thai",
        "tir_Ethi": "Tigrinya",
        "tsn_Latn": "Tswana",
        "tso_Latn": "Tsonga",
        "tur_Latn": "Turkish",
        "ukr_Cyrl": "Ukrainian",
        "urd_Arab": "Urdu",
        "urd_Latn": "Urdu (Romanized)",
        "uzn_Latn": "Northern Uzbek",
        "vie_Latn": "Vietnamese",
        "war_Latn": "Waray",
        "wol_Latn": "Wolof",
        "xho_Latn": "Xhosa",
        "yor_Latn": "Yoruba",
        "zho_Hans": "Chinese (Simplified)",
        "zho_Hant": "Chinese (Traditional)",
        "zsm_Latn": "Standard Malay",
        "zul_Latn": "Zulu",
    }
    
    return language_map.get(code, code)

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

def run_belebele(model_id, model_name, output_file, load_8bit, device=torch.device("cuda")):
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_8bit=load_8bit)
        if (not load_8bit):
            model = model.to(device)
    except Exception as e:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map='auto', load_in_8bit=load_8bit)
        if (not load_8bit):
            model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.eval()

    data = load_dataset("facebook/belebele")
    output = {
        "flores_passage": [],
        "question": [],
        "model": [],
        model_id + "_prob_A": [],
        model_id + "_prob_B": [],
        model_id + "_prob_C": [],
        model_id + "_prob_D": [],
        model_id + "_is_correct": [],
        model_id + "_picked_first_choice": [],
        model_id + "_picked_second_choice": [],
        model_id + "_picked_third_choice": [],
        model_id + "_picked_fourth_choice": [],
        "mc_answer1": [],
        "mc_answer2": [],
        "mc_answer3": [],
        "mc_answer4": [],
        "correct_answer_num": [],
        "language_code": [],
        "language_name": [],
        "link": [],
        "question_num": [],
    }
    alphas = ["(A)", "(B)", "(C)", "(D)"]

    for dataset in tqdm(data, desc="Processing all languages"):
        for i, example in tqdm(enumerate(data[dataset]) , desc="Processing " + dataset, leave=False, total=len(data[dataset])):
            context = example["flores_passage"]
            question = example["question"]
            choices = [example["mc_answer1"], example["mc_answer2"], example["mc_answer3"], example["mc_answer4"]]
            formatted_choices = "\n" + "\n".join([f"{alphas[i]} {choice}" for i, choice in enumerate(choices)])
            correct_answer = example["correct_answer_num"]

            formatted_prompt = prompt.format(context=context, question=question, choices=formatted_choices)

            losses = compute_negative_log_likelihoods(formatted_prompt, alphas, tokenizer, model, device=device)

            probs = torch.exp(losses * -1)
            probs = probs / torch.sum(probs)

            max_choice = torch.argmax(probs).item()

            is_correct = 1 if max_choice == (int(correct_answer) - 1) else 0

            output["flores_passage"].append(context)
            output["question"].append(question)
            output[model_id + "_prob_A"].append(probs[0].item())
            output[model_id + "_prob_B"].append(probs[1].item())
            output[model_id + "_prob_C"].append(probs[2].item())
            output[model_id + "_prob_D"].append(probs[3].item())
            output[model_id + "_is_correct"].append(is_correct)
            output["mc_answer1"].append(choices[0])
            output["mc_answer2"].append(choices[1])
            output["mc_answer3"].append(choices[2])
            output["mc_answer4"].append(choices[3])
            output["correct_answer_num"].append(int(correct_answer))
            output["language_code"].append(dataset)
            output["language_name"].append(convert_iso_code_to_language_name(dataset))
            output["link"].append(example["link"])
            output["question_num"].append(example["question_number"])

            # Keep track of which choice was picked
            output[model_id + "_picked_first_choice"].append(1 if max_choice == 0 else 0)
            output[model_id + "_picked_second_choice"].append(1 if max_choice == 1 else 0)
            output[model_id + "_picked_third_choice"].append(1 if max_choice == 2 else 0)
            output[model_id + "_picked_fourth_choice"].append(1 if max_choice == 3 else 0)
            output["model"].append(model_name)

            if i % 100 == 0:
                # Create the output directory if it doesn't exist
                if os.path.dirname(output_file) != "" and not os.path.exists(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file))
                
                # Write the results to the output directory
                df = pd.DataFrame(output)
                df.to_csv(output_file, index=False)

    # Create the output directory if it doesn't exist
    if os.path.dirname(output_file) != "" and not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    df = pd.DataFrame(output)
    df.to_csv(output_file, index=False)



def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    run_belebele(args.model_id, args.model_name, args.output_file, args.load_8bit)

if __name__ == "__main__":
    main()

# Example usage:
# python belebele.py --model_id gpt2 --output_file ./output/gpt2.csv