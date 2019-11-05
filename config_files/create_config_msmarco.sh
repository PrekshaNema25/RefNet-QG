#!bin/bash

output_dir=experiments/emnlp_19_msmarco/$1 
max_sequence_length=100 
embedding_dir=embeddings_msmarco_new/ 
random_seed=9091

### 	Question Encoder  ###
use_question_encoder=$2
word_concat=$2

####     Gate    ####
use_gate_policy=$3
use_rl_on_gate=$3
gate_loss_lambda=1.0
use_gated_context=$3

#####   Attn Diff  ####
use_attention_different=$4
use_attn_product=$4

mkdir -p $output_dir
cp experiments/emnlp_19/word_vanilla/config_final.ini $output_dir
sed -i "s#output_dir.*#output_dir = $output_dir#" "$output_dir/config_final.ini"
sed -i "s#random_seed.*#random_seed = $random_seed#" "$output_dir/config_final.ini"
sed -i "3s#max_sequence_length.*#max_sequence_length = $max_sequence_length#g" "$output_dir/config_final.ini"
sed -i "s#embedding_dir.*#embedding_dir= $embedding_dir#g" "$output_dir/config_final.ini"
sed -i "s#use_question_encoder.*#use_question_encoder = $use_question_encoder#g" "$output_dir/config_final.ini"
sed -i "s#use_gate_policy.*#use_gate_policy  = $use_gate_policy#g" "$output_dir/config_final.ini"
sed -i "s#use_rl_on_gate.*#use_rl_on_gate = $use_rl_on_gate#g" "$output_dir/config_final.ini"
sed -i "s#gate_loss_lambda.*#gate_loss_lambda = $gate_loss_lambda#g" "$output_dir/config_final.ini"
sed -i "s#use_gated_context.*#use_gated_context = $use_gated_context#g" "$output_dir/config_final.ini"
sed -i "s#use_attention_different.*#use_attention_different = $use_attention_different#g" "$output_dir/config_final.ini"
sed -i "s#use_attn_product.*#use_attn_product = $use_attn_product#g" "$output_dir/config_final.ini"
sed -i "s#word_concat.*#word_concat= $word_concat#g" "$output_dir/config_final.ini"
