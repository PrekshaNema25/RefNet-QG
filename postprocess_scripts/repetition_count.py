from nltk import ngrams
import sys 
import numpy as np

f = open(sys.argv[1])
ngrams_n = [1,2,3]
total_rep = []

for line in f.readlines():
	rep = []
	for n in ngrams_n:
		ngram_list = list(ngrams(line.rstrip().split(),n))
		ngram_set = set(ngram_list)
		fraction = (1 - float(len(ngram_set))/float(len(ngram_list)))*100.0
		rep.append(fraction)
	total_rep.append(rep)

rep_mean = np.mean(np.asarray(total_rep),axis=0)
for idx,i in enumerate(rep_mean):
	print ("%d-gram repetition %f percent"%(idx+1,i)) 
 
