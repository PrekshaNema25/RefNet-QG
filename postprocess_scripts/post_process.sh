for i in "data_500k_exp" "data_250k_exp" "data_50k_exp" "data_100k_exp"
do 
		python preprocess.py $i/stay
		python preprocess.py $i/nlb

done


e="_exp"
for i in "data_500k" "data_250k" "data_50k" "data_100k"
do
	python postprocess.py $i/test_content $i$e/nlb_results/results_b64_s256/test_final_results110_plabels $i$e/nlb/b64_s256/test_attention_weights110
	python postprocess.py $i/test_content $i$e/nlb_results/results_b64_s512/test_final_results110_plabels $i$e/nlb/b64_s512/test_attention_weights110
	python postprocess.py $i/test_content $i$e/nlb_results/results_b64_s128/test_final_results110_plabels $i$e/nlb/b64_s128/test_attention_weights110
	python postprocess.py $i/test_content $i$e/nlb_results/results_b32_s256/test_final_results110_plabels $i$e/nlb/b32_s256/test_attention_weights110
	python postprocess.py $i/test_content $i$e/nlb_results/results_b32_s512/test_final_results110_plabels $i$e/nlb/b32_s256/test_attention_weights110
	python postprocess.py $i/test_content $i$e/nlb_results/results_b32_s128/test_final_results110_plabels $i$e/nlb/b32_s256/test_attention_weights110

	python postprocess.py $i/test_content $i$e/stay_results/results_b64_s256/test_final_results110_plabels $i$e/stay/b64_s256/test_attention_weights110
	python postprocess.py $i/test_content $i$e/stay_results/results_b64_s512/test_final_results110_plabels $i$e/stay/b64_s512/test_attention_weights110
	python postprocess.py $i/test_content $i$e/stay_results/results_b64_s128/test_final_results110_plabels $i$e/stay/b64_s128/test_attention_weights110
	python postprocess.py $i/test_content $i$e/stay_results/results_b32_s256/test_final_results110_plabels $i$e/stay/b32_s256/test_attention_weights110
	python postprocess.py $i/test_content $i$e/stay_results/results_b32_s512/test_final_results110_plabels $i$e/stay/b32_s256/test_attention_weights110
	python postprocess.py $i/test_content $i$e/stay_results/results_b32_s128/test_final_results110_plabels $i$e/stay/b32_s256/test_attention_weights110

	zip -r results_$i.zip $i$e/nlb_results $i$e/stay_results
done 

