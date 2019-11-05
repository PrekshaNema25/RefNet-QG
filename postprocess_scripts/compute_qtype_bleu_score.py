import subprocess 
import signal 
import sys
import numpy as np
import os
from bleu_script.pycocoevalcap.eval_bleu_perline import main

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


stopwords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "over",  "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "too", "only", "myself", "those", "i", "after", "few", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "here", "than"]

what_words = ["is","was","stop","rest"]
what_lines = { "is":[],"was":[],"stop":[],"rest":[] }

question_words = {"what":0, "where":1, "when":2, "who":3, "how":4, "why":5,"which":6, "whom":7}
question_lines = {"what":[], "where":[], "when":[], "who":[], "how":[], "why":[],"which":[], "whom":[],"not_found":[]}


f = open(sys.argv[1])
g = open(sys.argv[2])

ref_lines = []
gen_lines = []

for ref_line,gen_line in zip(f.readlines(),g.readlines()):

	# print (ref_line,gen_line)
	found = 0
	ref_words = ref_line.lower().split(" ")
	for idx,word in enumerate(ref_words):

		for q_word in question_words:
			if q_word == word:
				found = 1
				question_lines[q_word].append((ref_line,gen_line)) 
				print (q_word,ref_line)
				if q_word == "what":

					if (ref_words[idx+1] == "is"):
						what_lines["is"].append((ref_line,gen_line))
					
					elif ref_words[idx+1] == "was":
						what_lines["was"].append((ref_line,gen_line))
					
					elif ref_words[idx+1] in stopwords:
						what_lines["stop"].append((ref_line,gen_line))

					else:
						what_lines["rest"].append((ref_line,gen_line))
	
	if found == 0:
		print ("not_found",ref_line)
		question_lines["not_found"].append((ref_line,gen_line)) 

f.close()
g.close()

bleus = {}
for q_word in question_lines:
	print (q_word)
	line1, line2 = zip(*question_lines[q_word])
	bleu = main(line1,line2)
	bleus[q_word] = bleu

what_bleus = {}
for word in what_lines:
	print (word)
	line1, line2 = zip(*what_lines[word])
	bleu = main(line1,line2)
	what_bleus[word] = bleu
	

print ("-"*30)
for idx,q_word in enumerate(question_lines):
	print ("%s: %0.3f %d" %(q_word,bleus[q_word], len(question_lines[q_word])) )

print ("-"*30)
for idx,word in enumerate(what_lines):
	print ("%s: %0.3f %d" %(word,what_bleus[word], len(what_lines[word])) )
