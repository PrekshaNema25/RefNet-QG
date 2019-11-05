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

_PATH="/scratch/scratch4/preksha/questiongeneration/GCN_QG/postprocess_scripts"
echo ${_PATH}
python /scratch/scratch4/preksha/questiongeneration/GCN_QG/postprocess_scripts/retreive.py $1/test_final_results$5
python ${_PATH}/retreive.py $1/valid_final_results$5



mkdir -p $1/predictions
mkdir -p $1/predictions/test
mkdir -p $1/predictions/valid
mkdir -p $1/plots
mkdir -p $1/plots/test
mkdir -p $1/plots/test



#cp $1/test_final_results_plabels $1/predictions/test/


#cp $1/valid_final_results_plabels $1/predictions/valid/

#cp $1/valid_final_results_plabels_copy $1/predictions/valid/
if [ "$3" == "True" ]
then
	python2 ${_PATH}/postprocess.py $2/test_content $1/test_final_results${5}_plabels $1/test_attention_weights$5
	python2 ${_PATH}/postprocess.py $2/valid_content $1/valid_final_results${5}_plabels $1/valid_attention_weights$5
	#python2 postprocess.py $2/valid_content $1/valid_final_results${5}_tlabels $1/valid_attention_weights$5

	#cp $1/test_final_results_plabels_copy $1/predictions/test
fi 

if [ "$4" == "True" ] 
then

	sed 's/<pad>//g' $2/test_field > $2/test_field_modified
	sed 's/<pad>//g' $2/valid_field > $2/valid_field_modified
	max_plots_save=100

	python2 ${_PATH}/plt_attention.py $1/test_awf $1/predictions/test/test_final_results_plabels_copy $2/test_field_modified $max_plots_save $1/plots/test
	python2 ${_PATH}/plt_attention.py $1/valid_awf $1/predictions/valid/valid_final_results_plabels_copy $2/valid_field_modified $max_plots_save $1/plots/valid

	rm $2/test_field_modified
	rm $2/valid_field_modified\

fi

python2 ${_PATH}/bleu_script/pycocoevalcap/eval_bleu_perline.py $2/test_summary  $1/test_final_results${5}_plabels_copy
python2 ${_PATH}/Answerability-Metric/answerability_score.py --data_type squad --ref_file $2/test_summary --hyp_file $1/test_final_results${5}_plabels_copy --ngram_metric Bleu_4  --re_weight 0.20 --ner_weight 0.41 --delta 0.66 --qt_weight 0.20

#python2 ${_PATH}/bleu_script/pycocoevalcap/eval_bleu_perline.py $2/test_summary  $1/test_final_results${5}_plabels

#echo "Validation Loss"
#python2 ../bleu_script/pycocoevalcap/eval_bleu_perline.py $1/valid_final_results${5}_tlabels  $1/valid_final_results${5}_plabels_copy
