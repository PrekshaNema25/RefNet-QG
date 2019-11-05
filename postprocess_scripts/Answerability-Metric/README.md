# Answerability-Metric
This repository contains the script to compute the questions based on the Answerability aspect. 
Towards a Better Metric for Evaluating Question Generation Systems - Preksha Nema and Mitesh Khapra ( EMNLP, 2018)

## Implementation
The script is a modified version of [coco-caption](https://github.com/tylin/coco-caption).

## Requirements
* `python setup.py install`
* Java
* pickle
* [NIST](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v13a.pl) (Optional)
* [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/) (Optional)

## Data Format
* The reference file and the hypothesis file should contain one question per line.
* NIST, METEOR scores should be precomputed using the given script, and should contain score for (ref_ques, hyp_ques) per line, corresponding to the ref_file, hyp_file.

## How to run the script
Run:
```bash
python answerability_score.py --data_type squad --ref_file examples/references.txt --hyp_file examples/hypotheses.txt --ner_weight 0.6 --qt_weight 0.2 --re_weight 0.1 --delta 0.7 --ngram_metric Bleu_3
```
* ngram_metric varies from [Bleu_1, Bleu_2, Bleu_3, Bleu_4, ROUGE_L, NIST, METEOR]

Note that the weights given to the stop_words is computed as (1 - ner_weight - qt_weight - re_weight) and the weight for the ngram metric is computed using (1 - delta)

## Testing
Run:
```bash
pip install -e '.[dev]'
pytest
```
