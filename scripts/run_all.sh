#bin/bash

echo "Running All Scripts"

echo "Running all experiment scripts"
./scripts/experiments/experiments.sh

echo "Running all postprocessing scripts"
./scripts/postprocessing/9-postprocessing.sh

echo "Running all analysis scripts"
./scripts/analysis/analysis.sh