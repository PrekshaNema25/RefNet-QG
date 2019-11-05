import random
import nltk
import numpy as np
import pickle
import sys
import copy
import os.path
import tensorflow as tf
from vocab import *
import scipy.sparse as sp
import codecs
import string

stopwords = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "over",  "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "too", "only", "myself", "those", "i", "after", "few", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

punct = [".",",","?","!","'",":",";","...","-","_","(",")","[","]"]


def question_categories(question_sent):
    question_sent_words = question_sent.strip().split(" ")
    label = []
    for  w in question_sent_words:
        if w.lower() in ["why", "whom", "where", "when", "how", "which", "what","who"]:
           label.append(0)
        elif w.lower() in stopwords or (len(w) == 1 and w in string.punctuation):
           label.append(1)
        elif len(w) > 0 and w[0].isupper():
           label.append(2)
        else:
           label.append(3)
    return label

class Datatype:

    def __init__(self, temp_dict):

        """ Defines the dataset for each category valid/train/test

        Args:
            name : Name given to this partition. For e.g. train/valid/test
            title: The summarized sentence for the given source document
            content: Source documents
            exm:  Number of examples in this partition
            max_length_content : Maximum length of source documents among all examples
            max_length_title: Maximum length of title among all examples
            
            global_count_train: pointer to retrieve the next batch during training
            global_count_test : pointer to retrieve the next batch during testing
        """

        self.name = temp_dict["name"]
        self.title_word = temp_dict["title"]
        
        self.content_word = temp_dict["content_word"]
        self.content_maxout_word = temp_dict["content_maxout_word"]
        self.content_maxout_vocab_word = temp_dict["content_maxout_vocab_word"]
        self.content_char = temp_dict["content_char"]
        
        self.content_dep_tree = temp_dict["content_dep_tree"]

        self.query_dep_tree = temp_dict["query_dep_tree"]
        self.query_word = temp_dict["query_word"]
        self.query_char = temp_dict["query_char"]
        self.query_position = temp_dict["query_position"]
        self.position_embeddings = temp_dict["position_embeddings"]

        self.number_of_examples = len(self.query_word)
        self.max_length_content = temp_dict["max_content"]
        self.max_length_query = temp_dict["max_query"]
        self.max_length_title = temp_dict["max_title"]
        self.max_length_title_prop = temp_dict["max_title_prop"]
        self.prop_indices = temp_dict["prop_indices"]
        self.title_pos = temp_dict["title_pos"]
        self.title_copy_word = temp_dict["title_copy"]
        self.question_label = temp_dict["question_label_encoded"]
        self.question_position = temp_dict["question_position_encoded"]
        self.qbleu_cat_encoded = temp_dict["qbleu_cat_encoded"]

        self.title_prop_labels = temp_dict["title_prop_labels"]
        print (self.name, " " , self.max_length_content, " " , self.max_length_title)
        self.global_count_train = 0
        self.global_count_test = 0

class PadDataset:

    def find_max_length(self, data, count, batch_size):

        """ Finds the maximum sequence length for data of 
            size batch_size

            Args:
                data: The data from which sequences will be chosen
                count: The pointer from which retrieval should be done.
                batch_size: Number of examples to be taken to find max.

        """
        data = data[count:count + batch_size]
        return max(len(data[i]) for i,_ in enumerate(data))

    def pad_data(self,data, max_length, pad_value=0):

        """ Pad the data to max_length given

            Args: 
                data: Data that needs to be padded
                max_length : The length to be achieved with padding

            Returns:
                padded_data : Each sequence is padded to make it of length
                              max_length.
        """

        padded_data = []
        sequence_length = []
        for lines in data:
            if (len(lines) < max_length):
                sequence_length.append(len(lines))
                temp = np.lib.pad(lines, (0,max_length - len(lines)),
                    'constant', constant_values=pad_value)
            else:
                temp = lines[:max_length]
                sequence_length.append(max_length)
            padded_data.append(temp)

        return padded_data, sequence_length


    def make_batch(self, data, batch_size,count, max_length,pad_value=0):

        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                count : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []

        batch = data[count:count+batch_size]
        count = count + batch_size

        if len(batch) < batch_size:
            batch = batch + data[:batch_size - len(batch)]
            count = batch_size - len(batch)

        batch, sequence_length  = self.pad_data(batch,max_length,pad_value=pad_value)

        return np.asarray(batch), sequence_length, count

    def make_batch_qu(self, data, batch_size, count):
        batch = []
        batch = data[count:count + batch_size] 
        if len(batch) < batch_size:
           batch = batch + data[:batch_size - len(batch)]
           count = batch_size - len(batch)
        return np.asarray(batch)

    def make_batch_position(self, data, batch_size, count, max_length_query, max_length_content):
        batch = []
        batch = data[count:count + batch_size]
        count = count + batch_size
        
        if len(batch) < batch_size:
            batch = batch + data[:batch_size - len(batch)]
            count = batch_size - len(batch)

        new_batch = []
        for line in batch:
            if max(line) > max_length_content:
                new_batch.append([0]*max_length_query)
            else:
                temp_line = line[:max_length_query]
                new_batch.append(np.pad(temp_line, (0, max_length_query - len(temp_line)), 'constant'))
        
        return np.asarray(new_batch), count

    def make_batch_mat(self, data,data_content,  batch_size, count, max_length):
        batch = []
        batch = data[count:count+batch_size]
        batch_new = data_content[count:count + batch_size]
        count = count +batch_size
        
        if len(batch) < batch_size:
            batch = batch + data[:batch_size - len(batch)]
            batch_new = batch_new + data_content[:batch_size - len(batch_new)]
            
            count = batch_size - len(batch)
        
        # print (batch,batch_new)
        # assert(len(batch_new) == len(batch)) 

        new_batch = []
        for ind, m in enumerate(batch):
            temp_mat = m.toarray().astype(np.float32)
            dim = len(batch_new[ind])
            temp_mat = temp_mat[0:dim, 0:dim]
            #temp_mat += np.transpose(temp_mat)

            temp_mat = (temp_mat >= 1).astype(np.float32)
            temp = np.sum(temp_mat, axis=1)
            temp = temp.flatten()
            degrees = np.diag(temp)
            degree_rt = np.linalg.inv(degrees)
            new_adj_mat = np.matmul(degree_rt, temp_mat)
            #new_adj_mat = np.matmul(new_adj_mat, degree_rt)
            if dim > max_length:
                new_adj_mat = new_adj_mat[:max_length, :max_length]
            else:
                new_adj_mat = np.pad(new_adj_mat, ((0, max_length - dim), (0, max_length - dim)), 'constant')
            
            new_batch.append(np.asarray(new_adj_mat))
        return np.asarray(new_batch), count
        
    def make_batch_char(self, data, batch_size,count, max_length_word, max_length_char):

        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                count : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []

        batch = data[count:count+batch_size]
        count = count + batch_size

        if len(batch) < batch_size:
            batch = batch + data[:batch_size - len(batch)]
            count = batch_size - len(batch)



        new_batch = []
        for line in batch:
            new_words = []
            for i, word in enumerate(line):
                if i >= max_length_word:
                    break
                if (len(word) <= max_length_char):
                    new_words.append(np.pad(word, (0, max_length_char - len(word)), 'constant'))
                else:
                    new_words.append(np.asarray(word[:max_length_char]))


            while len(new_words) < max_length_word:
                new_words.append(np.zeros(max_length_char, dtype=int))
                if (len(new_words) == max_length_word):
                    break
            new_batch.append(np.asarray(new_words))
        batch = np.asarray(new_batch)
        #print (batch)
        return batch, count

    
    def make_batch_position_embeddings(self, dt ,batch_size,count,max_length_content,pad_value=None,dim=100):

        pad_value = max_length_content
        position_embeddings, _, _ = self.make_batch(dt.position_embeddings,batch_size,count,max_length_content,pad_value=max_length_content)
        position_embeddings = np.asarray(position_embeddings)

        #embeddings = []

        #for i in range(dim//2):
        #    sin_embed = np.sin(position_embeddings/(5.0)**(2.0*i/dim))
        #    cos_embed = np.cos(position_embeddings/(5.0)**(2.0*i/dim))
        #    embeddings.extend([sin_embed,cos_embed])

        #position_embeddings = np.concatenate(embeddings,axis=-1)

        return np.reshape(position_embeddings, [-1,max_length_content])

    def next_batch(self, dt, batch_size, c=True):
        
        if (c is True):
            count = dt.global_count_train
        
        else:
            count = dt.global_count_test


        max_length_content = max(self.datasets[i].max_length_content for i in self.datasets)
        max_length_title   = max(self.datasets[i].max_length_title for i in self.datasets)
        max_length_query   = max(self.datasets[i].max_length_query for i in self.datasets)
        max_length_title_prop   = max(self.datasets[i].max_length_title_prop for i in self.datasets)

        contents_word, sequence_length_contents_word, count1 = self.make_batch(dt.content_word, batch_size,count, max_length_content)
        contents_maxout_words, _, _ = self.make_batch(dt.content_maxout_word, batch_size, count, max_length_content)
        contents_maxout_vocab_words, _, _ = self.make_batch(dt.content_maxout_vocab_word, batch_size, count, max_length_content )
        titles_word, sequence_length_titles_word,  _ = self.make_batch(dt.title_word, batch_size, count, max_length_title)

        titles_prop_word, sequence_length_titles_prop_word, _ = self.make_batch(dt.title_prop_labels, batch_size, count, max_length_title_prop)
        qbleu_cat, _, _ = self.make_batch(dt.qbleu_cat_encoded, batch_size, count, max_length_title)
 
        titles_copy_word, sequence_length_titles_word, _ = self.make_batch(dt.title_copy_word, batch_size, count, max_length_title)
        question_label  = self.make_batch_qu(dt.question_label, batch_size, count)
        question_position = self.make_batch_qu(dt.question_position, batch_size, count)
        title_pos, _, _ = self.make_batch(dt.title_pos, batch_size, count, max_length_title)
        prop_indices, _, _ = self.make_batch(dt.prop_indices, batch_size, count, max_length_title,pad_value=0)

        #title_prop_labels_batch, _, _ = self.make_batch(dt.title_prop_labels, batch_size, count, max_length_title, pad_value=4)

        # print ("*"*20,title_pos.shape)
        query_word, sequence_length_query_word, _ = self.make_batch(dt.query_word, batch_size, count, max_length_query)

        query_position,  _ = self.make_batch_position(dt.query_position, batch_size, count, max_length_query, max_length_content)

        if "use_positional_embeddings" in self.args["Embedding"] and self.args["Embedding"]["use_positional_embeddings"]:
            position_embeddings = self.make_batch_position_embeddings(dt,batch_size,count,max_length_content,pad_value=max_length_content,dim=int(self.args["Embedding"]["position_embeddings_dims"]))

        max_length_query_char = max_length_content_char = 20 
        contents_char,count_char = self.make_batch_char(dt.content_char, batch_size, count, max_length_content, max_length_content_char)
        query_char,  _ = self.make_batch_char(dt.query_char, batch_size, count, max_length_query, max_length_query_char)


        """
        contents_pos, _, _= self.make_batch(dt.content_pos, batch_size, count, max_length_content)
        titles_pos, _, _ = self.make_batch(dt.title_pos, batch_size, count, max_length_title)
        query_pos, _, _ == self.make_batch(dt.query_pos, batch_size, count, max_length_query)
        
        
        contents_ner, _, _= self.make_batch(dt.content_ner, batch_size, count, max_length_content)
        query_ner, _, _ == self.make_batch(dt.query_ner, batch_size, count, max_length_query)
        """
        contents_dep_tree, count_adj_mat = self.make_batch_mat(dt.content_dep_tree, dt.content_word,  batch_size, count, max_length_content)
        query_dep_tree, _ = self.make_batch_mat(dt.query_dep_tree, dt.query_word, batch_size, count, max_length_query)
        #assert(count_char == count1 == count_adj_mat)

        weights = copy.deepcopy(titles_word).astype(np.float32)

        for i in range(titles_word.shape[0]):
            for j in range(titles_word.shape[1]):
                if (weights[i][j] == 0):
                        weights[i][j] = 0
                elif weights[i][j] in titles_prop_word[i]:
                        weights[i][j] = 1
		else:
                        weights[i][j] = 1
			
        prop_word_weights = copy.deepcopy(contents_word)
        weights_prop = copy.deepcopy(titles_prop_word).astype(np.float32)

        for i in range(titles_prop_word.shape[0]):
            for j in range(titles_prop_word.shape[1]):
                if (weights_prop[i][j] != 1 and weights_prop[i][j] in self.stopwords_encoded):
                        weights_prop[i][j] = float(self.args["Aux_Loss"]["stop_word_weight"])
                elif (weights_prop[i][j] > 0):
                        weights_prop[i][j] = 1
                else:
                        weights_prop[i][j] = 0

        content_weights = copy.deepcopy(contents_word)
        for i in range(contents_word.shape[0]):
            for j in range(contents_word.shape[1]):
                if contents_word[i][j] in query_word[i]:
                        content_weights[i][j] = 1
                if (content_weights[i][j] > 0):
                        content_weights[i][j] = 1
                elif content_weights[i][j] == 0:
                        content_weights[i][j] = 0
        content_weights = content_weights.astype(np.float32)
        if (c == True): 
            dt.global_count_train = count1 % dt.number_of_examples
        else:
            dt.global_count_test = count1 % dt.number_of_examples
        
        encoder_inputs = {}
        decoder_inputs = {}
        query_inputs = {}

        encoder_inputs["word"] = contents_word
        encoder_inputs["char"] = contents_char
        encoder_inputs["maxout_word"] =contents_maxout_words
        encoder_inputs["maxout_vocab_word"] = contents_maxout_vocab_words

        if "use_positional_embeddings" in self.args["Embedding"] and self.args["Embedding"]["use_positional_embeddings"]:
            encoder_inputs["positional"] = position_embeddings
        else:
            encoder_inputs["positional"] = []

 
        encoder_inputs["content_weights"] = content_weights
        query_inputs["word"] = query_word
        query_inputs["char"] = query_char


        if self.args["Hyperparams"]["use_pos_decoder"] == "True":
            decoder_inputs["pos"]  = title_pos
        
        else:
            decoder_inputs["pos"]  =  []

        if self.args["GCN"]["use_passage_gcn"] == "True":
            encoder_inputs["dep_tree"] = contents_dep_tree
        
        else:
            encoder_inputs["dep_tree"] = []
        
        if self.args["GCN"]["use_query_gcn"] == "True":
            query_inputs["dep_tree"] = query_dep_tree

        else: 
            query_inputs["dep_tree"] = []


        query_inputs["position"] = query_position
        
        encoder_inputs["seq_length"] = sequence_length_contents_word
        query_inputs["seq_length"] = sequence_length_query_word
        
        decoder_inputs["word"] = titles_word
        decoder_inputs["label_switch"] = titles_copy_word
        decoder_inputs["weights"] = weights
        decoder_inputs["question_label"] = question_label
        decoder_inputs["question_position"] = question_position
        decoder_inputs["property_word"]= titles_prop_word
        decoder_inputs["property_weights"] = weights_prop
        decoder_inputs["prop_indices"] = prop_indices
	
        #print ("Next Batch Weights Prop",weights_prop)
	#print ("Next Batch Weights",weights)
	#print ("Next Batch word",titles_word)
	#print ("Next Batch property word",titles_prop_word)
        #print (prop_indices)
        #print (contents_maxout_words)
        #print (contents_word)
        #print(contents_maxout_vocab_words)
        #print (encoder_inputs, query_inputs, decoder_inputs)
        return encoder_inputs, query_inputs, decoder_inputs
    
    def load_data_file(self,name, title_prop_file, title_file, content_file, query_file, global_max_content = 10000, global_max_query = 10000, global_max_title = 10000, global_max_title_prop=1000):

        qbleu_cat_encoded = []
        title = codecs.open(title_file,'r', encoding='utf-8').readlines()
        content = codecs.open(content_file,'r', encoding='utf-8').readlines()
        query= codecs.open(query_file, 'r', encoding="utf-8").readlines()
        title_prop_labels = codecs.open(title_prop_file, 'r', encoding="utf-8").readlines()

        for line in title:
            qbleu_cat_encoded.append(question_categories(line))

        title = [i.lower() for i in title]
        content = [i.lower() for i in content]
        query = [i.lower() for i in query]
        if self.args["Hyperparams"]["use_same_prop_grammar"] == "True":
		title_prop_labels = copy.deepcopy(title)
	else:
	    	title_prop_labels = []
	    	for title_lines in title:
	    	    title_prop_words = []
	    	    title_words = title_lines.split(" ")
	    	    for title_word in title_words:
	    		if title_word not in stopwords + punct:
	    	    	    title_prop_words.append(title_word) 
	    	    title_prop_labels.append(" ".join(title_prop_words))				

        if self.args["Hyperparams"]["use_pos_decoder"] == "True":
            title_pos = codecs.open(title_pos_file, 'r', encoding="utf-8").readlines()
        
        else:
            title_pos = []


        if self.args["GCN"]["use_passage_gcn"] == "True":
            
            content_dep_tree = pickle.load(codecs.open(content_file + "_sparse_adj_mats.pkl", "rb", errors='ignore'), encoding = 'latin1')
            content_dep_tree = [sp.csr_matrix(i) for i in content_dep_tree]
        else:
            content_dep_tree = []
            
        if self.args["GCN"]["use_query_gcn"] == "True":

            query_dep_tree = pickle.load(codecs.open(query_file + "_sparse_adj_mats.pkl", "rb", errors='ignore'), encoding='latin1')
            query_dep_tree = [sp.csr_matrix(i) for i in query_dep_tree]
        else:
            query_dep_tree = []
        
        #if self.args["Encoder"]["use_property_attention"] == "True":
        #   title_prop_labels = pickle.load(codecs.open(title_prop_file, "rb", errors="ignore"))
        #   title_prop_labels = [[4] + i + [3] for i in title_prop_labels]
        #else:
        #   title_prop_labels = []

        title_encoded = []
        title_pos_encoded = []
        content_pos_encoded = []
        content_ner_encoded = []
        query_pos_encoded = []
        query_ner_encoded = []
        content_encoded = []
        content_maxout_encoded = []
        content_maxout_vocab_encoded = []
        content_char_encoded = []
        query_char_encoded = []
        query_encoded = []
        sequence_length_list = []
        question_label_encoded = []
        question_position_encoded = []
        title_pos_encoded = []
        title_prop_labels_encoded = []

        question_words = {"what":0, "where":1, "when":2, "who":3, "how":4, "why":5,"which":6, "whom":7}
 
        max_title = 0
        for lines in title:
            temp = [self.vocab.encode_word(word) for word in lines.lower().strip().split()]
            if (len(temp) > max_title):
                max_title = len(temp) + 2
            title_encoded.append([self.vocab.encode_word("<s>")] + temp + [self.vocab.encode_word("<eos>")])

        max_title_prop = 0 
        for lines in title_prop_labels:
            temp = [self.vocab.encode_word(word) for word in lines.lower().strip().split()]
            if (len(temp) > max_title_prop):
                max_title_prop = len(temp) + 2
            title_prop_labels_encoded.append([self.vocab.encode_word("<s>")] + temp + [self.vocab.encode_word("<eos>")])

        for lines in title_pos:
            temp = [self.vocab.encode_pos(word) for word in lines.strip().split()]
            if (len(temp) > max_title):
                max_title = len(temp) + 2
            title_pos_encoded.append([self.vocab.encode_pos("<s>")] + temp + [self.vocab.encode_pos("<eos>")])

        max_content = 0
        for lines in content:
            temp_maxout_dict = {}
            temp = [self.vocab.encode_word(word) for word in lines.lower().strip().split()]
            count = 0
            for i in temp:
                if i not in temp_maxout_dict:
                    temp_maxout_dict[i] = count
                    count+=1
            temp_maxout_sent = [temp_maxout_dict[i] for i in  temp]
            if (len(temp) > max_content):
                max_content = len(temp)
            content_encoded.append(temp)
            content_maxout_encoded.append(temp_maxout_sent)
            unique_words = []
            temp_maxout_vocab = []
            for i in temp:
                if i not in unique_words:
                    temp_maxout_vocab.append(i)
                    unique_words.append(i)
            content_maxout_vocab_encoded.append(temp_maxout_vocab)


        max_query = 0
        for lines in query:
            temp = [self.vocab.encode_word(word) for word in lines.lower().strip().split()]
            if (len(temp) > max_query):
                max_query = len(temp)
            query_encoded.append(temp)
        
        for lines in content:
            words_list = []
            for words in lines.lower().strip().split():
                temp_words = [self.vocab.encode_char(w) for w in words]
                words_list.append(temp_words)
            content_char_encoded.append(words_list)

        for lines in query:
            words_list = []
            for words in lines.lower().strip().split():
                temp_words = [self.vocab.encode_char(w) for w in words]
                words_list.append(temp_words)
            query_char_encoded.append(words_list)

        query_position_encoded = []
        position_embeddings = []

        #print (len(content), len(query))
        for lines_content, lines_query in zip(content, query):

            words_list_content = lines_content.strip().split(" ")
            words_list_query = lines_query.strip().split(" ")
            found = 0

            #position_embedding_line = [max_content]*len(words_list_content)
            position_embedding_line = [0]*len(words_list_content)

            for ind in (i for i, e in enumerate(words_list_content) if e == words_list_query[0]):
                if " ".join(words_list_content[ind:ind + len(words_list_query)]) == " ".join(words_list_query):
                    query_position_encoded.append([ j for j in range(ind, ind + len(words_list_query))])
                    
                    for j in range(ind, ind + len(words_list_query)):
                        position_embedding_line[j] = 1 
                    #for idx,j in enumerate(range(ind + len(words_list_query), len(words_list_query))):
                    #    position_embedding_line[j] = 0
                    #for idx,j in enumerate(range(ind-1,-1,-1)):
                    #    position_embedding_line[j] = -1*(idx+1)
                    found = 1
                    break

            position_embeddings.append(position_embedding_line)

            if  found == 0:
                query_position_encoded.append([0]* len(words_list_query))
            if len(query_position_encoded[-1]) == 0 : print (found, query_position_encoded[-1], words_list_query, words_list_content)

        assert (len(query_position_encoded) == len(query_encoded))
        
        title_copy_encoded = []
        for lines_content, lines_title in zip(content, title):

            words_line_content = lines_content.strip().split(" ")
            words_list_title = lines_title.strip().split(" ")
            temp_title = []
            for word in words_list_title:
               if True:
                  temp_title.append(1)#(self.vocab.encode_word(word))
               elif word in words_line_content:
                  temp_title.append(0)#(self.vocab.len_vocab + words_line_content.index(word))
               else:
                  temp_title.append(1)#(self.vocab.encode_word(word))
            title_copy_encoded.append([1] + temp_title + [1])

        for i in title:
            words_list_title = i.strip().split(" ")
            found = False
            for ind,w in enumerate(words_list_title):
                if w.lower() in question_words:
                   question_label_encoded.append(question_words[w.lower()])
                   question_position_encoded.append(ind)
                   found = True
                   break
            if found == False:
               question_label_encoded.append(8)
               question_position_encoded.append(0)
            
        print (name) 
        
        prop_indices = []
	for quest,props in zip(title,title_prop_labels):
	        quest = quest.rstrip().split(" ")
	        props = props.rstrip().split(" ")
                quest = ["<s>"] + quest + ["<eos>"]
                props = ["<s>"] + props + ["<eos>"]
	        indices  = []
	        for idx,word in enumerate(quest):
	                index = []
	                found = 0
	                for idx_prop, word_prop in enumerate(props):
	                        if word == word_prop:
	                                index.append((idx_prop,abs(idx-idx_prop)))
	                                found = 1
	                if found == 1:
	                        indices.append(sorted(index,key=lambda x:x[1])[0][0])
	                else:
	                        indices.append(int(self.args["Decoder"]["max_prop_steps"])-1)
                prop_indices.append(indices)
                #print ("Quest is ", quest)
                #print ("Props is ", props)
                #print ("Indices is", indices)
	
        temp_dict = {"name": name, 
                    "title":title_encoded, 
                    "title_copy":title_copy_encoded,
                    "title_pos" : title_pos_encoded,
                    "title_prop_labels" : title_prop_labels_encoded,
                    "content_word":content_encoded,
                    "content_maxout_word":content_maxout_encoded,
                    "content_maxout_vocab_word": content_maxout_vocab_encoded,
		            "prop_indices":prop_indices,
                    "query_word":query_encoded, 
                    "content_char":content_char_encoded,
                    "query_char":query_char_encoded,
                     "query_position": query_position_encoded,
                     "prop_indices": prop_indices,
                     "position_embeddings": position_embeddings,
                     "content_dep_tree":content_dep_tree,
                     "query_dep_tree":query_dep_tree,
                    "question_label_encoded":question_label_encoded,
                    "question_position_encoded":question_position_encoded,
                    "qbleu_cat_encoded":qbleu_cat_encoded, 
                    "max_content": max_content if (global_max_content > max_content) else global_max_content,
                    "max_title": max_title if (global_max_title > max_title)  else global_max_title,
                    "max_title_prop": max_title_prop if (global_max_title_prop > max_title)  else global_max_title_prop, 
                    "max_query":  global_max_query}

        return Datatype(temp_dict)


    def load_data(self, wd="../Data/", max_content=1000, max_query=1000, max_title=1000, max_title_prop=1000):

        s = wd
        self.datasets = {}
        for i in ("train", "valid", "test"):
            temp_t = s + i + "_summary"
            temp_v = s + i + "_content"
            temp_f = s + i + "_query"
            temp_prop_t = s + i + "_summary_prop_labels"

            self.datasets[i] = self.load_data_file(i, temp_prop_t, temp_t, temp_v, temp_f, max_content, max_query, max_title, max_title_prop)


    def __init__(self, args,working_dir = "../Data/", embedding_size=100, vocab_length = 100, global_count = 0, embedding_dir = "../Data",
        max_content = 1000, max_query = 1000, max_title = 1000, max_title_prop=1000):

        filenames = [working_dir + "train_summary" , working_dir + "train_content", working_dir + "train_query"]
        pos_dir = working_dir + "train_pos_summary"
        self.args = args
        self.global_count = 0
        self.vocab = Vocab(vocab_length)
        self.vocab.construct_vocab(args,filenames, pos_dir, embedding_size, vocab_length, embedding_dir)
        self.load_data(working_dir, max_content, max_query, max_title, max_title_prop)
	self.stopwords_encoded = [self.vocab.encode_word(i) for i in stopwords + punct]
	print ("stop words encoded",self.stopwords_encoded)
    def length_vocab(self):
        return self.vocab.len_vocab


    def decode_to_sentence(self, decoder_states):
        s = ""
        for temp in (decoder_states):
            if temp not in self.vocab.index_to_word:
                    word = "<unk>"
            else:
                word = self.vocab.decode_word(temp)
            s = s + " " + word

        return s
    
    def decode_to_pos_tags(self, decoder_states):
        
        s = ""
        for temp in (decoder_states):
            if temp not in self.vocab.index_to_pos:
                    pos = "<unk>"
            else:
                pos = self.vocab.decode_pos(temp)
            s = s + " " + pos
        return s

def main():
    x = PadDataset([sys.argv[1],sys.argv[2]])
    x.load_data()
    print (x.decode_to_sentence([2,1,2,3,4,5,8,7]))
    for i in range(0,10):
        x.next_batch(x.datasets["train"], 2,True)
    x.load_data()
    print (x.decode_to_sentence([2,1,2,3,4,5,8,7]))
    for i in range(0,10):
        x.next_batch(x.datasets["train"], 2,True)
        x.next_batch(x.datasets["train"],2, False)

if __name__ == '__main__':
    main()
