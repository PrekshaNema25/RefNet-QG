from basic_predict_script import *
import sys, time
import os, json
import argparse
from os.path import basename
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def _load_model(model_dir):
    """
    """
    model = PredictOnWeatherData(model_dir = model_dir)
    print('Loading the model [%s]'%(model_dir))
    return model

def _load_input_output(filename, delimiter='$'):
    records = []
    summaries = []
    
    f=open(filename,'r')
    for line in f:
        s = line.strip().split(delimiter)
        # print s[1:-1]
        # print [en.replace(',','') for en in s[1:-1]]
        # print s[-1]
        # time.sleep(20)
        records.append([en.replace(',','') for en in s[1:-1]])
        summaries.append(s[-1])
    f.close()
    
    return records,summaries


def loadJsonToMap(json_file):
    data = json.load(open(json_file))
    imgToAnns = {}
    for entry in data:
        if entry['image_id'] not in imgToAnns.keys():
                imgToAnns[entry['image_id']] = []
        summary = {}
        summary['caption'] = entry['caption']
        summary['image_id'] = entry['caption']
        imgToAnns[entry['image_id']].append(summary)
    return imgToAnns

def _evaluate_model(ref_file, model_dirs, save_dir):
    #model_dirs = open(model_file).read().split('\n')
    #models = [m for m in model_dirs if len(m) > 2]

    for modelname in model_dirs:
       print('Calculating bleu for: ' + modelname)
       model_file_name = basename(modelname)

       res_json = os.path.join(save_dir, model_file_name + '_out.json')
       ref_json = ref_file # reference file
       coco = loadJsonToMap(ref_json)
       cocoRes = loadJsonToMap(res_json)
       cocoEval = COCOEvalCap(coco, cocoRes)
       cocoEval.params['image_id'] = cocoRes.keys()
       cocoEval.evaluate()
       # create output dictionary
       out = {}
       for metric, score in cocoEval.eval.items():
           out[metric] = score
       # serialize to file, to be read from Lua
       json.dump(out, open(os.path.join(save_dir, model_file_name + 'results.json'), 'w'))

def evalModels(ref_file, pre_file):
   res_json = pre_file
   ref_json = ref_file # reference file
   coco = loadJsonToMap(ref_json)
   cocoRes = loadJsonToMap(res_json)
   cocoEval = COCOEvalCap(coco, cocoRes)
   cocoEval.params['id'] = cocoRes.keys()
   cocoEval.evaluate()
   # create output dictionary
   out = {}
   for metric, score in cocoEval.eval.items():
       out[metric] = score
   # serialize to file, to be read from Lua
   json.dump(out, open(os.path.join('results.json'), 'w'))

def _get_json_format(o):
    '''
    Format to be written in json file. Excepted by coco lib
    '''
    data_pred = []
    for id, s in enumerate(o):
        line = {}
        line['id'] = id
        line['caption'] = s
        data_pred.append(line)
    return data_pred


def calculate_scores(predicted, reference):

    p_file = open(predicted, "r")
    r_file = open(reference, "r")

    pred_sents = []
    ref_sents = []

    for line in p_file :
        pred_sents.append(line)

    for line in r_file:
        ref_sents.append(line)

    p_res = _get_json_format(pred_sents)
    r_res = _get_json_format(ref_sents)

    ref_file = os.path.join('reference.json')
    json.dump(r_res, open(ref_file, 'wb'), indent = 4)

    pre_file = os.path.join('predicted.json')
    json.dump(p_res, open(pre_file, 'wb'), indent = 4)

    evalModels(ref_file, pre_file)

def _run(model_dir, encoded_records):
    model = _load_model(model_dir)
    outputs, _ = model.predict(encoded_records, True)
    return outputs


def _run_and_evaluate_models(model_file, dumps_file, save_dir):
    records, summaries = _load_input_output(dumps_file)
    j_summaries = _get_json_format(summaries)
    ref_file = os.path.join(save_dir, 'reference.json')
    json.dump(j_summaries, open(ref_file, 'wb'), indent = 4)

    #model_dirs = open(model_file).read().split('/n')
    model_dirs = open(model_file).read().split('\n')
    model_dirs = [m for m in model_dirs if len(m) > 2]

    for model_dir in model_dirs:
        #if len(model_dir) < 2:
        #continue # to aviod last line problem
        o = _run(model_dir, records)
        print o
        o = [l.split('<EOS>')[0] for l in o]
        data_pred = _get_json_format(o)
        model_file_name = basename(model_dir)
        print model_file_name
        json.dump(data_pred, open(os.path.join(save_dir, model_file_name + '_out.json'), 'wb'), indent = 4)
    _evaluate_model(ref_file, model_dirs, save_dir)


if __name__ == '__main__':
    calculate_scores(sys.argv[1], sys.argv[2])