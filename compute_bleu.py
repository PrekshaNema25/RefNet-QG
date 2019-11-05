from nlgeval import compute_metrics
import sys
from nltk.tokenize import word_tokenize
import codecs
import re
print (sys.argv[1], sys.argv[2])

x = codecs.open(sys.argv[1], encoding="utf-8")
x = x.readlines()
y = codecs.open(sys.argv[2], encoding="utf-8")
y = y.readlines()

x = [" ".join(word_tokenize(i)) for i in x]
y = [" ".join(word_tokenize(i)) for i in y]

x_w = codecs.open(sys.argv[1], "w", encoding="utf-8")
for i in x:
    x_w.write(i.strip().lower() + " \n")
y_w = codecs.open(sys.argv[2], "w", encoding="utf-8")

for i in y:
    y_w.write(i.strip() + "\n")

x_w.close()
y_w.close()
print (compute_metrics(sys.argv[1], [sys.argv[2]]))
x = x = codecs.open(sys.argv[1], encoding="utf-8")
x = x.readlines()
x_w = codecs.open(sys.argv[1], "w", encoding="utf-8")
for i in x:
    x_w.write(i.strip() + "\n")

