#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Evaluation Method for summarization tasks, including BLUE and ROUGE score
Visualization of Attention Mask Matrix: plot_attention() method
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import os
import sys
import numpy as np
import matplotlib

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # drawing heat map of attention weights

plt.rcParams['font.sans-serif'] = ['SimSun']  # set font family

import time


def plot_attention(data, itr , X_label=None, Y_label=None, directory_name=None,epoch = 0):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(25, 20))  # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)

    # Set axis labels
    if X_label != None and Y_label != None:
        X_label = [x_label.decode('utf-8') for x_label in X_label]
        Y_label = [y_label.decode('utf-8') for y_label in Y_label]

        xticks = range(0, len(X_label))
        # xticks = [x + 0.5 for x in xticks]
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels(X_label, minor=False, rotation=30, fontsize=20)  # labels should be 'unicode'

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(Y_label, minor=False, fontsize=20)  # labels should be 'unicode'

        print (X_label, Y_label)
        ax.grid(True)

    # Save Figure
    # plt.title(u'Attention Heatmap')
    timestamp = int(time.time())
    file_name = directory_name + '/attention_heatmap_' + str(epoch) + '_' + str(itr) + ".jpg"
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure

def save_plot(awf,y_label_file_name,x_label_file_name,max_count,directory_name,epoch=0):

    x_label_file = open(x_label_file_name)
    y_label_file = open(y_label_file_name)

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    x_range = range(40)
    x_label = []
    for lines in x_label_file:
        lines = lines.decode('utf-8')
        x_label.append(lines.split())

    y_label = []
    for lines in y_label_file:
        y_label.append(lines.split())

    count = 0
    print ("Done", len(awf), len(x_label), len(y_label))
    for i in range(len(awf)):

        data_per_batch = np.squeeze(np.asarray(awf[i]))
        data_per_batch = np.transpose(data_per_batch, axes=[1,0,2])
	
	print (np.shape(data_per_batch))
        for j in range(len(data_per_batch)):

		if (count > max_count):
			break

                x_range = range(40)
		data_current = data_per_batch[j]
                data_current = data_current[:len(y_label[count])]
		data_current = np.transpose(data_current)
		#print ("the shape", np.shape(data_current), len(x_label))
		data_current = data_current[:len(x_label[count])]

		data_current = np.transpose(data_current)

		x_range =  x_range[1:len(y_label[count])]
                #print (np.shape(data_current))

		fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.set_xticklabels(x_label[count], minor=False, rotation=20)

		plot_attention(data_current, count, x_label[count], y_label[count], directory_name, epoch)
		count = count + 1 

if __name__ == "__main__":

    awf = pickle.load(open(sys.argv[1]))
    max_count = int(sys.argv[4])
    directory_name = sys.argv[5]
    save_plot(awf,sys.argv[2],sys.argv[3],max_count,directory_name)
