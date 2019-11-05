from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
#from cider.cider import Cider
import json
import sys
import os
import codecs
import pickle
import sys
reload(sys)
import numpy as np 
sys.setdefaultencoding('utf8')

def _get_json_format(pred_sents, name):
    '''
    Format to be written in json file. Excepted by coco lib
    '''
    #p_file = codecs.open(o, "r", encoding="utf-8", errors="replace")


    #pred_sents = []

    #for line in p_file :
    #    pred_sents.append(line)
    data_pred = []
    for id, s in enumerate(pred_sents):
        line = {}
        line['image_id'] = id
        line['caption'] = s
        data_pred.append(line)

    ref_file = os.path.join(name)
    json.dump(data_pred, codecs.open(ref_file, 'w', encoding="utf-8", errors="ignore"), indent = 4)
    #json.dump(data_pred, open(ref_file, 'w'), indent=4)
    return name

def loadJsonToMap(json_file):
    data = json.load(codecs.open(json_file, "r", encoding="utf-8", errors="ignore"))
    #data = json.load(open(json_file, "r"))
    imgToAnns = {}
    print (len(data))
    for entry in data:
        #print entry['image_id'],entry['caption']
        if entry['image_id'] not in imgToAnns.keys():
                imgToAnns[entry['image_id']] = []
        summary = {}
        summary['caption'] = entry['caption']
        summary['image_id'] = entry['caption']
        imgToAnns[entry['image_id']].append(summary)
    return imgToAnns

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.keys()}


    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco[imgId]#.imgToAnns[imgId]
            res[imgId] = self.cocoRes[imgId]#.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(cBleu(4), ["cBleu_1", "cBleu_2", "cBleu_3", "cBleu_4"]),
            #(Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L")
            #(Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print "%s: %0.3f"%(m, sc) 
	            bleu = sc
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs() 
        return bleu

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        x = []
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
    
def main(ref_sents,pred_sents):

       	ref_sents = [i.lower() for i in ref_sents]
       	pred_sents = [i.lower() for i in pred_sents]
       	Z = list(zip(ref_sents, pred_sents))
       	scores = []
         
       	file_1 = _get_json_format(ref_sents[:len(Z)], 'reference')
       	file_2 = _get_json_format(pred_sents[:len(Z)], 'predicted')
       	coco = loadJsonToMap(file_1)
       	cocoRes = loadJsonToMap(file_2)
       	cocoEval = COCOEvalCap(coco, cocoRes)
       	cocoEval.params['image_id'] = cocoRes.keys()
       	bleu = cocoEval.evaluate() 
        return bleu

if __name__ == '__main__':
	main(open(sys.argv[1]).readlines(),open(sys.argv[2]).readlines())
