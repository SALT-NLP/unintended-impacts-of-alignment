#bin/bash

# Postprocess all of the outputs into nice results (runs on cpu)

python postprocessing/join-csv.py --directory  ./outputs/md3-game/ --output_file ./results/2_md3game/md3game.csv
    
python postprocessing/join-csv.py --directory  ./outputs/belebele/ --output_file ./results/3_belebele/belebele.csv --belebele --drop_columns flores_passage,question,prob_A,prob_B,prob_C,prob_D,,mc_answer1,mc_answer2,mc_answer3,mc_answer4,link,question_num

python postprocessing/tydiqa.py --input_dir ./outputs/tydiqa-goldp --output_file ./results/4_tydiqa/goldp.csv --gold_dir ./data/tydiqa/tydiqa-goldp-v1.1-dev/ --output_list

python postprocessing/join-csv.py --directory  ./outputs/globalopinionsqa --output_file ./results/6_globalopinionsqa/globalopinionsqa.csv