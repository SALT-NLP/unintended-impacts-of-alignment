# Unintended Impacts of LLM Alignment on Global Representation
Official Repository for the ACL 2024 Paper: [Unintended Impacts of LLM Alignment on Global Representation](https://arxiv.org/abs/2402.15018)

![Figure 1: Countries are highlighted according to the reward assigned by the starling reward model.  Read caption below.  USA and Australia are highly preferred, while countries in the Middle East and Africa are dispreferred.](rewards.png)

Figure 1: Country rewards for Starling 7B Reward Model prompted with "User: Where are you from? Assistant: I am from {country}." Starling assigns higher rewards to English-speaking Western nations and lower rewards to countries in the Middle East/Africa.

# TLDR
This repository contains all the code for the ACL 2024 Paper [Unintended Impacts of LLM Alignment on Global Representation](https://arxiv.org/abs/2402.15018).  If you are looking for the AskRedditCountries dataset check out our [huggingface]().

This repository covers all the steps to reproduce the results in our paper exactly.  We also include all the intermediate/final results in the `/outputs/`, `/results/`, and `/visualization/` folders.

If you want to reproduce all experiments in our paper run the following bash script:

```
TODO: add in a single bash script that will run absolutely everything
```

# Table of Contents
 - [Installation](#installation)
 - [Experiments](#experiments)
 - [Process Outputs to Results](#process-outputs-to-results)
 - [Process Results to Visuals](#process-results-to-visuals)
 - [Contact](#contact)
 - [Citation](#citation)

# Installation

```
conda create -n "alignment-impacts" python=3.11.5 ipython
conda activate alignment-impacts
pip install -r requirements.txt
```

# Experiments

To run all experiments run the following script
```
TODO: Single Experiment Script
```

## Ask Starling Where its From
Run the "Where From" script
```
./scripts/experiments/0-where_from_reward_model.sh
```

## Dialect Intent Detection
First download the md3 dataset following the instructions in `/data/md3/md3/README.txt`, [here](data/md3/md3/README.txt).

Next run the data cleaning script
```
./scripts/experiments/1-md3_clean.sh 
```

Now you are set to run the md3 experiment script
```
./scripts/experiments/2-md3_experiments.sh
```

This will write the outputs to `./outputs/md3-game/`.

## Belebele Reading Comprehension

Run the Belebele Reading Comprehension script
```
./scripts/experiments/3-belebele_experiments.sh
```

## TyDiQA Question Answering
Run the TyDiQA Question Answering script
```
./scripts/experiments/4-tydiqa_experiments.sh
```

## Ultrachat and Tulu SFT Language ID
Run the Language ID script
```
./scripts/experiments/5-langid_experiments.sh
```

## Global Opinions QA
Run the Global Opinions QA script
```
./scripts/experiments/6-globalopinions_experiments.sh
```

## Ask Reddit Rewards
Run the Ask Reddit Country Opinions Reward Modeling script
```
./scripts/experiments/7-askreddit-rewards.sh
```

# Process Outputs to Results
Run the postprocessing script
```
./scripts/postprocessing/8-postprocessing.sh
```

This will take the outputs from `./outputs/` and process them into single csv files in the `./results/` directory

# Process Results to Visuals

## Where from cloropleth
Run the "Where From" analysis script


# Contact
**Michael Ryan**: [Scholar](https://scholar.google.com/citations?user=8APGEEkAAAAJ&hl=en) | [Twitter](http://twitter.com/michaelryan207) | [Github](https://github.com/XenonMolecule) | [LinkedIn](https://www.linkedin.com/in/michael-ryan-207/) | [Research Gate](https://www.researchgate.net/profile/Michael-Ryan-86) | [Personal Website](http://michryan.com/) | [michaeljryan@stanford.edu](mailto://michaeljryan@stanford.edu)

# Citation
If you use this code or our AskRedditCountries dataset please cite our paper:
```
@misc{ryan2024unintended,
      title={Unintended Impacts of LLM Alignment on Global Representation}, 
      author={Michael J. Ryan and William Held and Diyi Yang},
      year={2024},
      eprint={2402.15018},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
