#!bin/bash

output_dir=experiments/emnlp_19_msmarco/$1 
output_dir_rl=experiments/emnlp_19_msmarco/${1}_rl 

### RL
rl_plus_crossent=True
use_rl_plus_entropy=True
max_epochs=3

mkdir -p $output_dir_rl
cp ${output_dir}/config_final.ini $output_dir_rl
cp ${output_dir}/best_model.* $output_dir_rl

echo "best model copied"
sed -i "s#output_dir.*#output_dir = $output_dir_rl/#" "$output_dir_rl/config_final.ini"
sed -i "s#rl_plus_crossent.*#rl_plus_crossent= $rl_plus_crossent#g" "$output_dir_rl/config_final.ini"
sed -i "s#use_rl_plus_entropy.*#use_rl_plus_entropy= $use_rl_plus_entropy#g" "$output_dir_rl/config_final.ini"
sed -i "s#max_epochs.*#max_epochs= $max_epochs#g" "$output_dir_rl/config_final.ini"
