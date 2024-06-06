#bin/bash

echo "Running all experiment scripts"

echo "Running 0-where_from_reward_model.sh"
./scripts/experiments/0-where_from_reward_model.sh

echo "Running 1-md3_clean.sh"
./scripts/experiments/1-md3_clean.sh

echo "Running 2-md3_experiments.sh"
./scripts/experiments/2-md3_experiments.sh

echo "Running 3-belebele_experiments.sh"
./scripts/experiments/3-belebele_experiments.sh

echo "Running 4-tydiqa_experiments.sh"
./scripts/experiments/4-tydiqa_experiments.sh

echo "Running 5-langid_experiments.sh"
./scripts/experiments/5-langid_experiments.sh

echo "Running 6-global_opinions_experiments.sh"
./scripts/experiments/6-global_opinions_experiments.sh

echo "Running 7-ask_reddit_experiments.sh"
./scripts/experiments/7-ask_reddit_experiments.sh

echo "Running 8-ask_reddit_reward_model.sh"
./scripts/experiments/8-ask_reddit_reward_model.sh