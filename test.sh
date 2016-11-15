#!/bin/bash
# usage: ./test.sh caption
echo "input caption: "$@
python pre-process.py word2idx_mapping.txt "$1" 2> temp
idx_vec=$(<temp)
# echo $idx_vec
th main.lua -data 52k.hdf5 -warm_start_model ./results/20161108_0911_model_6.t7 -test_only 1 -input "$idx_vec" 
rm temp
# python preprocess.py custom glove.840B.300d.txt --train ./data/score_sentiment_queries.201611031637.txt.train --custom_name 52k
# python preprocess.py custom glove.840B.300d.txt --train ./data/score_sentiment_queries.201611031637.txt.train --custom_name 52k --test test_52k