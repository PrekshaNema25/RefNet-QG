#!/bin/bash

# Clean the extracted labels by removing the <pad> and <eos> symbols.
# Post process the labels to also generate the copy predictions
# Save the attention weight plots

#Usage:
#$1 : output_dir
#$2 : data_dir
#$3 : If post process copy is required
#$4 : If attention plots need to be plotted.
#sh extract_labels $output_dir $data_dir

#_PATH=
_PATH="/home/ubuntu/scratch/TheOneWhereThingsAreWorking/postprocess_scripts"
echo ${_PATH}
python ${_PATH}/retreive.py $1/test_prev_final_results$5
python ${_PATH}/retreive.py $1/valid_final_results$5



mkdir -p $1/predictions
mkdir -p $1/predictions/test
mkdir -p $1/predictions/valid
mkdir -p $1/plots
mkdir -p $1/plots/test
mkdir -p $1/plots/test



#cp $1/test_prev_final_results_plabels $1/predictions/test/


#cp $1/valid_final_results_plabels $1/predictions/valid/

#cp $1/valid_final_results_plabels_copy $1/predictions/valid/
if [ "$3" == "True" ]
then
        python2 ${_PATH}/postprocess.py $2/test_content $1/test_prev_final_results${5}_plabels $1/test_prev_attention_weights$5
        #python2 ${_PATH}/postprocess.py $2/valid_content $1/valid_final_results${5}_plabels $1/valid_attention_weights$5
        #python2 postprocess.py $2/valid_content $1/valid_final_results${5}_tlabels $1/valid_attention_weights$5

        #cp $1/test_prev_final_results_plabels_copy $1/predictions/test
fi

sleep 1

