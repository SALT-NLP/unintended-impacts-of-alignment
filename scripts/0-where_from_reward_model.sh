#bin/bash

# Run the reward testing
# Ask a Reward Model where it's from!
cd ./code

model="berkeley-nest/Starling-RM-7B-alpha"
model_name="starling-rm-7b-alpha"

python reward-testing.py --reward_batch_size 4 --seed 221 --output_file ../outputs/reward-testing/where_from_$model_name.csv

cd ../