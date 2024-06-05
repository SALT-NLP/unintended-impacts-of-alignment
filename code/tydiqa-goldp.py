import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import torch
import numpy as np
from util.util import load_models
from transformers import GenerationConfig, DataCollatorForLanguageModeling
from collections import defaultdict
import json
import os
from tqdm import tqdm
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="Compute scores for TyDi QA GoldP")

    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="The name of the model to use.",
    )
    parser.add_argument(
        "--output_dir",
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
        "--num_shots",
        type=int,
        default=0,
        help="The number of shots to use for few-shot learning"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="The batch size to use for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=221,
        help="The random seed to use"
    )
    parser.add_argument(
        "--greedy",
        action='store_true',
        default=False,
        help="Whether or not to use greedy decoding"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The top p value to use for top p sampling, only used if greedy is false"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature to use for sampling, only used if greedy is false"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use for beam search"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers to use for multiprocessing"
    )
    parser.add_argument(
        "--error_file",
        type=str,
        default="errors.txt",
        help="The file to output errors to"
    )

    args = parser.parse_args()

    return args

INSTRUCTION = """Please answer the following questions about the text below by extracting the relevant answer from the context. 

Use the following format:
Context: A passage containing the answer to the question.
Question: The question being asked.
Extracted Answer: The answer to the question using a direct excerpt from the context.

"""

EXAMPLE = """Context: {context}
Question: {question}
Extracted Answer: {answer}

"""

PREANSWER = """Context: {context}
Question: {question}
Extracted Answer:"""

def get_training_dataset():
    language_groups = {}
    train_dataset = load_dataset("tydiqa", "secondary_task", split="train")
        
    for example in train_dataset:
        example_id = example["id"]
        language = example_id.split("-")[0]
        
        if language not in language_groups:
            language_groups[language] = []
        
        language_groups[language].append(example)
    
    return language_groups

def compute_tydiqa_goldp_extractions(model_id:str, output_dir:str, load_8bit:bool, num_shots:int=0, eval_batch_size:int=32, greedy:bool=True, top_p:float=1.0, temperature:float=1.0, num_beams:int=1, num_workers:int=1, error_file:str="errors.txt"):
    
    val_dataset = load_dataset("tydiqa", "secondary_task", split="validation")

    id_map = {}
    id_list = []
    lang_map = {}
    lang_list = []
    for example in val_dataset:
        if not example["id"] in id_map:
            id_map[example["id"]] = len(id_map)
            id_list.append(example["id"])
        language = example["id"].split("-")[0]
        if not language in lang_map:
            lang_map[language] = len(lang_map)
            lang_list.append(language)

    accelerator = Accelerator()

    language_groups = {}
    if (num_shots > 0):
        # Load the training dataset if we are using few-shot learning
        language_groups = get_training_dataset()

    model, tokenizer = load_models(model_id, load_8bit)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def construct_prompts(example):
        batch = {"input_ids": [], "attention_mask": [], "language": "", "id": ""}
        
        example_id = example["id"]
        language = example_id.split("-")[0]
        fs_examples = []
        if (num_shots > 0):
            fs_examples = random.sample(language_groups[language], num_shots)
        prompt = [INSTRUCTION]

        # Add few-shot examples to the prompt
        for ex in fs_examples:
            context = ex["context"]
            question = ex["question"]
            answer = ex["answers"]["text"][0]
            prompt.append(EXAMPLE.format(context=context, question=question, answer=answer))
        prompt.append(PREANSWER.format(context=example["context"], question=example["question"]))
        
        batch["id"] = id_map[example_id]
        batch["language"] = lang_map[language]

        tokenized = tokenizer("".join(prompt), truncation=True, padding=True, return_tensors="pt")
        batch["input_ids"] = tokenized["input_ids"][0]
        batch["attention_mask"] = tokenized["attention_mask"][0]

        return batch

    with accelerator.main_process_first():
        val_dataset = val_dataset.map(
            construct_prompts,
            batched=False,
            num_proc=num_workers,
            remove_columns=["context", "question", "answers", "title"],
            load_from_cache_file=False,
            desc="Constructing prompts",
        )

    # No longer need training data for constructing few shot examples so free up memory
    del language_groups

    generation_config = GenerationConfig.from_pretrained(model_id, num_beams=num_beams, do_sample=(not greedy), top_p=top_p, temperature=temperature, num_return_sequences=1, max_new_tokens=50) # max_length=tokenizer.model_max_length

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers)

    model, val_loader = accelerator.prepare(model, val_loader)

    model.eval()

    results = defaultdict(dict)
    # Loop over the val dataloader
    progress_bar = tqdm(range(len(val_loader)), desc="Evaluating", disable=not accelerator.is_local_main_process)
    for example in val_loader:
        with torch.no_grad():
            language = example["language"]
            example_id = example["id"]

            # Feed the examples to the generate function for the model using the generation config
            outputs = None
            try:
                outputs = accelerator.unwrap_model(model).generate(input_ids=example["input_ids"], attention_mask=example["attention_mask"], generation_config=generation_config)
            except RuntimeError as e:
                torch.cuda.empty_cache()
                try:
                    outputs = accelerator.unwrap_model(model).generate(input_ids=example["input_ids"], attention_mask=example["attention_mask"], generation_config=generation_config)
                except RuntimeError as e2:
                    print(e2)
                    with open(error_file, "a") as f:
                        for i in range(len(example_id)):
                            print("Skipping example", id_list[example_id[i]])
                            results[lang_list[language[i]]][id_list[example_id[i]]] = ""
                            f.write("Skipping example from language" + lang_list[language[i]] + ", ID: " + id_list[example_id[i]] + "\n")
                    if accelerator.is_local_main_process:
                        progress_bar.update(1)
                    continue

            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            outputs = accelerator.gather_for_metrics(outputs).cpu().numpy()

            # Save the results by language and associate the id with the generated output
            for i, output in enumerate(outputs):
                out = tokenizer.decode(output[len(example["input_ids"][i]):], skip_special_tokens=True)
                results[lang_list[language[i]]][id_list[example_id[i]]] = out

            if accelerator.is_local_main_process:
                progress_bar.update(1)

    if accelerator.is_main_process:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the results to the output directory
        for language, language_results in results.items():
            with open(os.path.join(output_dir, language + ".json"), "w") as f:
                json.dump(language_results, f)
        

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    compute_tydiqa_goldp_extractions(args.model_id, args.output_dir, args.load_8bit, args.num_shots, args.eval_batch_size, args.greedy, args.top_p, args.temperature, args.num_beams, args.num_workers, args.error_file)

if __name__ == "__main__":
    main()