import pickle
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats as s
x = x = pickle.load(open(sys.argv[1],"rb"))

ner,re,sw,qt,d, pred, true = [],[],[],[],[],[],[]

for i in x:
	ner.append(i['w_ner'])
	qt.append(i['w_qt'])
	sw.append( i['w_sw'])
	re.append(i['w_re'])
	d.append(i['d'])

print ("Ner, qt, sw, re, d", np.mean(ner), np.mean(qt), np.mean(sw), np.mean(re) , np.mean(d))
print ("Ner, qt, sw, re, d", np.std(ner), np.std(qt), np.std(sw), np.std(re) , np.std(d))
print ((np.std(ner) + np.std(qt) + np.std(sw) +  np.std(re) + np.std(d))/5)



all_scores = pickle.load(open(sys.argv[2],"rb"))

def get_answerability_score(all_scores, ner_weight, qt_weight, re_weight, d, ngram_metric="Bleu_4"):
	print(len(all_scores))
	ref_scores = [x['ref'] for x in all_scores]
	fluent_scores = [x[ngram_metric] for x in all_scores]
	imp_scores =  [x['imp'] for x in all_scores]
	qt_scores = [x['qt'] for x in all_scores]
	sw_scores = [x['sw'] for x in all_scores]
	ner_scores =  [x['ner'] for x in all_scores]

	new_scores = []

	for i in range(len(imp_scores)):
	    answerability = re_weight*imp_scores[i] + ner_weight*ner_scores[i]  + \
		    qt_weight*qt_scores[i] + (1-re_weight - ner_weight - qt_weight)*sw_scores[i]

	    temp = d*answerability + (1-d)*fluent_scores[i]
	    new_scores.append(temp)
	    print ("New Score:{} Ner Score: {} RE Score {} SW Score {} QT Score {} ".format(temp, ner_scores[i], imp_scores[i], sw_scores[i], qt_scores[i]))

	print ("Mean Answerability Score Across Questions: {} N-gram Score: {}".format(np.mean(fluent_scores), np.mean(new_scores)))
	np.savetxt(os.path.join(args.output_dir , 'ngram_scores.txt'), fluent_scores)
	np.savetxt(os.path.join(args.output_dir , 'answerability_scores.txt'),new_scores)
"""
a = ref_scores
#noise = np.random.normal(0,0.03, len(ref_scores))
#ref_scores += noise

 #new_scores #fluent_scores
#noise = np.random.normal(0,0.03, len(ref_scores))
#b += noise

fluent_scores = new_scores
b = fluent_scores
lr = linear_model.LinearRegression().fit(np.reshape(a,(-1,1)), np.reshape(fluent_scores,(-1,1)) )
b_new = lr.predict(np.reshape(a,(-1,1)))
plt.plot(ref_scores ,b,'o', c='#1f77b4')
plt.plot(a, b_new, color='red')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.ylabel(sys.argv[4],fontsize=15)
plt.xlabel(sys.argv[5],fontsize=15)
plt.title(sys.argv[6],fontsize=18)
plt.show()
#np.savetxt('results/new_scores_image_' +sys.argv[2], new_scores)
"""
f = open("fluent_scores_quora","w")
for i in fluent_scores:
	f.write(str(i) + "\n")
print ("Old correlation is", s.pearsonr(fluent_scores, ref_scores), s.spearmanr(fluent_scores, ref_scores))
print ("New correlation is", s.pearsonr(new_scores, ref_scores), s.spearmanr(new_scores, ref_scores))
