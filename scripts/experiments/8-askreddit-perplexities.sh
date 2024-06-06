#bin/bash

# Run the tydiqa experiments
cd ./code

# Test the script on cpu (this can take many hours, you should kill after a few test iterations)
# microsoft/Phi-3-mini-4k-instruct
# model="microsoft/Phi-3-mini-4k-instruct"
# model_name="Phi-3-mini-4k-instruct_8bit"
# python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# We recommend running the other models on a GPU (a single A6000 was sufficient for us)

# Llama-2-7b-hf
model="meta-llama/Llama-2-7b-hf"
model_name="Llama-2-7b-hf_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Llama-2-7b-chat-hf
model="meta-llama/Llama-2-7b-chat-hf"
model_name="Llama-2-7b-chat-hf_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Mistral-7b-v0.1
model="mistralai/Mistral-7B-v0.1"
model_name="Mistral-7B-v0.1_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Mistral-7b-sft-beta
model="HuggingFaceH4/mistral-7b-sft-beta"
model_name="Mistral-SFT-7B_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# OpenChat-3.5
model="openchat/openchat_3.5"
model_name="OpenChat3.5_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Qwen1.5-7B-Chat
model="Qwen/Qwen1.5-7B-Chat"
model_name="qwen1.5_chat_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Qwen1.5-7B
model="Qwen/Qwen1.5-7B"
model_name="qwen1.5_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Starling-LM-7B-alpha
model="berkeley-nest/Starling-LM-7B-alpha"
model_name="starling-alpha-7b_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Tulu-2-DPO
model="allenai/tulu-2-dpo-7b"
model_name="Tulu-2-DPO_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Tulu-2-SFT
model="allenai/tulu-2-7b"
model_name="Tulu-2-SFT_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Yi-6B-Chat
model="01-ai/Yi-6B-Chat"
model_name="yi-6b-chat_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Yi-6B
model="01-ai/Yi-6B"
model_name="yi-6b_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

# Zephyr-7B-beta
model="HuggingFaceH4/zephyr-7b-beta"
model_name="zephyr-7b-beta_8bit"
python ask-reddit-perplexity-testing.py --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-perplexity-testing/$model_name-reformatted.csv --add_article --model_id $model --reformat_response --use_chat_template

cd ../