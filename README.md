# Let's Ask Again: Refine Network for Automatic Question Generation




## Requirements:
* python2.7
* [tensorflow-1.14]
* [gensim](https://pypi.python.org/pypi/gensim)
* [nltk](http://www.nltk.org/install.html)
* [matplotlib](https://matplotlib.org/users/installing.html)

## Data Download:
    SQuAD (v1) 
    The dataset was preprocessed using nltk word_tokenizer and was lower cased. The dataset is store in the following format:
    data/
	train_content (passage/sentence which contains the answer) 
        train_query   ( answer)
        train_summary ( question needs to be generated)    
    
        valid_content 
        valid_query
        valid_summary

        test_content
        test_query
        test_summary
 
    The test split corresponds to dev split in all datasets. The valid split is 10\% of the train dataset, 
    selected randomly .

## Pretrained Embeddings
    Download the pretrained embeddings for a given language and then store it to a folder squad_embeddings/
    * English Embedding: bash extract_embeddings.sh english embeddings/
    
    
## Proposed Model: Training
    * To run only the RefNet-QG model on SQuAD model:
      python run_model_training.py ../data/ config_files/config_file_sentence.ini

    
 ## Proposed Models: Inference
    
    * The syntax for running the inference is also similar to running the training 
      python run_model_inference.py ../data/ config_files/config_file_sentence.ini
   
 
 
 ## Post Processing Scripts: Copy Mechanism and Generate Attention Weight Plots:
     cd postprocess_scripts
     
     # sh extract_labels_plots.sh <output_dir> <data_directory> <epoch_number>
     sh extract_labels_beam.sh  ../output ../data True False 1000

* [nlgeval] (https://github.com/Maluuba/nlg-eval)     
* [Bleu_4](https://github.com/tylin/coco-caption) 
* [ROUGE_4](https://github.com/gregdurrett/berkeley-doc-summarizer/blob/master/rouge/ROUGE/ROUGE-1.5.5.pl)
* [NIST_4](ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-v13.pl)

