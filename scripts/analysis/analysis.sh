#bin/bash

echo "Running all analysis scripts"

echo "Running 10-where_from_chloropleth.sh"
./scripts/analysis/10-where_from_chloropleth.sh

echo "Running 11-md3_game_analysis.sh"
./scripts/analysis/11-md3_game_analysis.sh

echo "Running 12-belebele_analysis.sh"
./scripts/analysis/12-belebele_analysis.sh

echo "Running 13-tydiqa_analysis.sh"
./scripts/analysis/13-tydiqa_analysis.sh

echo "Running 14-langid.sh"
./scripts/analysis/14-langid.sh

echo "Running 15-global_opinions.sh"
./scripts/analysis/15-global_opinions.sh

echo "Running 16-ask_reddit_chloropleth.sh"
./scripts/analysis/16-ask_reddit_chloropleth.sh

echo "Running 17-ask_reddit_correlation.sh"
./scripts/analysis/17-ask_reddit_correlation.sh