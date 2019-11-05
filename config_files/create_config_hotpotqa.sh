#!bin/bash

output_dir=experiments/emnlp_19_hotpotqa/$1 
output_dir_rl=experiments/emnlp_19_hotpotqa/${1}_rl 
embedding_dir=embeddings_hotpotqa/ 

#encoder
max_sequence_length=150 
beam_size=3

# decoder
max_decoder_steps=40
max_prop_steps=40

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

### RL
rl_plus_crossent=True
use_rl_plus_entropy=True
max_epochs=3

mkdir -p $output_dir
mkdir -p $output_dir_rl


cp experiments/emnlp_19/word_vanilla/config_final.ini $output_dir
sed -i "s#output_dir.*#output_dir = $output_dir/#" "$output_dir/config_final.ini"
sed -i "3s#max_sequence_length.*#max_sequence_length = $max_sequence_length#g" "$output_dir/config_final.ini"

sed -i "35s#max_sequence_length.*#max_sequence_length = $max_decoder_steps#g" "$output_dir/config_final.ini"
sed -i "s#max_decoder_steps.*#max_decoder_steps = $max_decoder_steps#g" "$output_dir/config_final.ini"
sed -i "s#max_prop_steps.*#max_prop_steps = $max_prop_steps#g" "$output_dir/config_final.ini"
sed -i "s#beam_size.*#beam_size = $beam_size#g" "$output_dir/config_final.ini"


sed -i "s#embedding_dir.*#embedding_dir= $embedding_dir#g" "$output_dir/config_final.ini"
sed -i "s#use_question_encoder.*#use_question_encoder = $use_question_encoder#g" "$output_dir/config_final.ini"
sed -i "s#use_gate_policy.*#use_gate_policy  = $use_gate_policy#g" "$output_dir/config_final.ini"
sed -i "s#use_rl_on_gate.*#use_rl_on_gate = $use_rl_on_gate#g" "$output_dir/config_final.ini"
sed -i "s#gate_loss_lambda.*#gate_loss_lambda = $gate_loss_lambda#g" "$output_dir/config_final.ini"
sed -i "s#use_gated_context.*#use_gated_context = $use_gated_context#g" "$output_dir/config_final.ini"
sed -i "s#use_attention_different.*#use_attention_different = $use_attention_different#g" "$output_dir/config_final.ini"
sed -i "s#use_attn_product.*#use_attn_product = $use_attn_product#g" "$output_dir/config_final.ini"
sed -i "s#word_concat.*#word_concat= $word_concat#g" "$output_dir/config_final.ini"

cp ${output_dir}/config_final.ini $output_dir_rl
cp ${output_dir}/best_model* $output_dir_rl
sed -i "s#output_dir.*#output_dir = $output_dir_rl/#" "$output_dir_rl/config_final.ini"
sed -i "s#rl_plus_crossent.*#rl_plus_crossent= $rl_plus_crossent#g" "$output_dir_rl/config_final.ini"
sed -i "s#use_rl_plus_entropy.*#use_rl_plus_entropy= $use_rl_plus_entropy#g" "$output_dir_rl/config_final.ini"
sed -i "s#max_epochs.*#max_epochs= $max_epochs#g" "$output_dir_rl/config_final.ini"
