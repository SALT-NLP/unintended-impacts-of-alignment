#bin/bash

# Run the askreddit reward testing
cd ./code

#!/bin/bash
cd /nlp/scr/mryan0/dialectal-llms/code/

model="berkeley-nest/Starling-RM-7B-alpha"
model_name="starling-rm-7b-alpha"

# Just answer with the country name
# python ask-reddit-reward-testing.py --reward_batch_size 1 --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-reward-testing/$model_name.csv --add_article --reward_model $model

# Answer in the rephrased format
python ask-reddit-reward-testing.py --reward_batch_size 1 --seed 221 --input_file ../data/askreddit_countries/AskRedditCountries_final.csv --output_file ../outputs/ask-reddit-reward-testing/$model_name-reformatted.csv --add_article --reward_model $model --reformat_response

cd ../