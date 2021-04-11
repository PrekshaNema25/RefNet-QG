from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
import os.path
import time
import math
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from optparse import OptionParser

from utils.dataset_iterator_two_decoder import *
from utils.model import *
from postprocess_scripts.plt_attention import save_plot
from utils.make_model_post_first_decoder import *
import os
import configparser
import pickle
import subprocess
import signal
from collections import OrderedDict
from utils.hypothesis import *
from nlgeval import compute_metrics

def optimistic_restore(session, save_file,quora=False):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    if quora:
        saved_shapes_new = {}
        for i in saved_shapes:
            saved_shapes_new["quora/"+i] = saved_shapes[i]
        saved_shapes = saved_shapes_new

    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
			if var.name.split(':')[0] in saved_shapes])
    restore_vars = {}
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
    	for var_name, saved_var_name in var_names:
    		curr_var = name2var[saved_var_name]
    		var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    if quora:
                        saved_var_name = saved_var_name[6:]
                restore_vars[saved_var_name] = (curr_var)            
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def count_parameters():

    print ("--- Variable List ---")
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        
        print (variable.name,variable_parameters)
        sys.stdout.flush()
        total_parameters += variable_parameters
    return total_parameters 

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

class run_model:

    def __init__(self, wd, config_filename, inference = False):

        """ The model is initializer with the hyperparameters.

            Args:
                wd : working data directory
                config : Config() obeject for the hyperparameters.
        """

        # Use default hyperparameters
        print("DEBUG: In init function")

        self.config  = configparser.ConfigParser(strict=False)
        self.config.read(config_filename)
        self.new_lr = float(self.config["Hyperparams"]["learning_rate"])
        self.rl_lambda_value = float(self.config["Hyperparams"]["rl_lambda_value"])
        self.num_gpus = 2

        if inference:
            print("In INFERENCE" + "-"*30)
            self.config["Hyperparams"]["batch_size"] = str(int(self.config["BeamSearch"]["batch_size"])) #*int( self.config["BeamSearch"]["beam_size"]))
            print("BATCH SIZE",self.config["Hyperparams"]["batch_size"])


        # Vocabulary and datasets are initialized.
        self.dataset = PadDataset(self.config,wd, int(self.config["Embedding"]["word_embedding_size"]), vocab_length = int(self.config["Embedding"]["word_vocab_length"]), 
                                  embedding_dir = self.config["Embedding"]["embedding_dir"], max_content = int(self.config["Encoder"]["max_sequence_length"]),
                                  max_query = int(self.config["Query"]["max_sequence_length"]), max_title=int(self.config["Decoder"]["max_sequence_length"]),
                                  max_title_prop=int(self.config["Decoder"]["max_prop_steps"]))

        print("DEBUG: Init Exit")

    def add_placeholders(self,name=""):

        """ Generate placeholder variables to represent input tensors
        """

        self.encode_input_word_placeholder  = tf.placeholder(tf.int32, shape=(None, int(self.config["Encoder"]["max_sequence_length"])), name ='encode'+name)
        self.encode_input_maxout_word_placeholder  = tf.placeholder(tf.int32, shape=(None, int(self.config["Encoder"]["max_sequence_length"])), name ='encode_maxout'+name)
        self.encode_input_maxout_vocab_word_placeholder  = tf.placeholder(tf.int32, shape=(None, int(self.config["Encoder"]["max_sequence_length"])), name ='encode_maxout_vocab'+name)
        self.query_input_word_placeholder   = tf.placeholder(tf.int32, shape=(None, int(self.config["Query"]["max_sequence_length"])), name='query'+name)
        self.decode_input_word_placeholder  = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_sequence_length"])-1),   name = 'decode'+name)
        self.encode_input_content_placeholder = tf.placeholder(tf.float32, shape=(None, int(self.config["Encoder"]["max_sequence_length"])), name="encoder_weights"+name)
        self.encode_input_char_placeholder  = tf.placeholder(tf.int32, shape=(None, int(self.config["Encoder"]["max_sequence_length"]), int(self.config["Embedding"]["max_word_length"])), name ='encode_char'+name)
        self.query_input_char_placeholder   = tf.placeholder(tf.int32, shape=(None, int(self.config["Query"]["max_sequence_length"]), int(self.config["Embedding"]["max_word_length"])), name='query_char'+name)
        self.encode_sequence_length  = tf.placeholder(tf.int32, shape=(None,), name ='encode_length'+name)
        self.query_sequence_length  = tf.placeholder(tf.int32, shape=(None,), name='query_length'+name)
        self.decode_sequence_length  = tf.placeholder(tf.int32, shape=(None,),   name = 'decode_length'+name)
        self.label_placeholder         = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_sequence_length"])-1),   name = 'labels'+name)
        self.prop_label_placeholder         = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_sequence_length"])-1),   name = 'labels_props'+name)
        self.weights_placeholder = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_sequence_length"])-1), name="weights"+name)
        self.question_label_placeholder = tf.placeholder(tf.int32, shape=(None,), name="question_labels"+name)
        self.question_position_placeholder = tf.placeholder(tf.int32, shape=(None,), name="question_position_labels"+name)
        self.encode_adj_matrix = tf.placeholder(tf.float32, shape=(None, int(self.config["Encoder"]["max_sequence_length"]),int(self.config["Encoder"]["max_sequence_length"])), name= "passage_dep_tree"+name)
        self.query_adj_matrix = tf.placeholder(tf.float32, shape=(None, int(self.config["Query"]["max_sequence_length"]), int(self.config["Query"]["max_sequence_length"])), name="query_dep_tree"+name)
        self.query_position_placeholder = tf.placeholder(tf.int32, shape=(None, int(self.config["Query"]["max_sequence_length"])), name="query_position"+name)
        self.encoder_positional_embedding = tf.placeholder(tf.float32, shape=(None, int(self.config["Encoder"]["max_sequence_length"]), int(self.config["Embedding"]["position_embeddings_dims"])), name="encoder_positional"+name)
        #self.decode_input_pos_placeholder = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_sequence_length"])-1), name="pos_title"+name)
        #self.label_pos_placeholder = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_sequence_length"])-1), name="pos_label"+name)
        self.prev_decoder_label_placeholder = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_prop_steps"])-1), name="prop_decoder_labels"+name)
        self.prev_decoder_input_placeholder = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_prop_steps"])-1), name="prop_decoder_words"+name)
        self.weights_decoder_label_placeholder = tf.placeholder(tf.float32, shape=(None, int(self.config["Decoder"]["max_prop_steps"])-1), name="prop_decoder_weights"+name)
        self.prop_indices_placeholder = tf.placeholder(tf.int32, shape=(None, int(self.config["Decoder"]["max_sequence_length"])-1), name = "prop_indices"+name)
        self.feed_previous_placeholder = tf.placeholder(tf.bool, name='feed_previous'+name)
        self.feed_true_placeholder = tf.placeholder(tf.bool, name='feed_previous_true'+name)
        self.is_sample_placeholder = tf.placeholder(tf.bool, name='is_sample'+name)
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name="learning_rate"+name)
        self.dropout_rate_placeholder = tf.placeholder(tf.float32, name="dropout"+name)

    def run_inference(self):

        """ Train the graph for a number of epochs 
        """
        with tf.Graph().as_default():


            tf.set_random_seed(int(self.config["Hyperparams"]["random_seed"]))

            len_vocab = self.dataset.length_vocab()
            char_vocab  = self.dataset.vocab.len_vocab_char
            embedding_initializer = self.dataset.vocab.embeddings

            self.add_placeholders()

            self.model  = make_model(self.config, embedding_initializer=embedding_initializer,
                                    vocab_length=len_vocab, vocab_char_length=char_vocab, vocab_dict=self.dataset.vocab, dropout_param=self.dropout_rate_placeholder)
            self.model.is_sample = False
            # Build a Graph that computes predictions from the inference model.
            self.encoder_input_batch = {"word": self.encode_input_word_placeholder,
                                    "maxout_word" : self.encode_input_maxout_word_placeholder,
                                    "maxout_vocab_word": self.encode_input_maxout_vocab_word_placeholder,
                                   "char": self.encode_input_char_placeholder,
                                   "seq_length": self.encode_sequence_length,
                                   "positional": self.encoder_positional_embedding,
                                   "dep_tree":self.encode_adj_matrix, 
                                   "content_weights": self.encode_input_content_placeholder, 
                                   "dropout": self.dropout_rate_placeholder
                                   }
            self.decoder_input_batch = {"word":self.decode_input_word_placeholder, 
                                   "labels":self.label_placeholder,
                                   "property_word": self.prev_decoder_input_placeholder,
                                   "property_labels": self.prev_decoder_label_placeholder,
                                   "content_prop_weights": self.weights_decoder_label_placeholder,
                                    #"labels_pos" : self.label_pos_placeholder,
                                    #"pos" : self.decode_input_pos_placeholder
                                    }
            self.query_input_batch = {"word": self.query_input_word_placeholder,
                                   "char": self.query_input_char_placeholder,
                                   "seq_length": self.query_sequence_length,
                                   "dep_tree": self.query_adj_matrix,
                                   "position":self.query_position_placeholder}



            self.decoder_cell = []
            self.dropout_param = 1
            with tf.variable_scope("Decoder_Copy_Cell", reuse=tf.AUTO_REUSE):
                self.decoder_cell.append(self.model.model.select_cell( "decoder_cell", self.config["Decoder"]["cell_type"], int(self.config["Decoder"]["hidden_size"]), 1, dropout_param=self.dropout_param)[0])

            with tf.variable_scope("Decoder_Copy_Cell_Second", reuse=tf.AUTO_REUSE):
                self.decoder_cell.append(self.model.model.select_cell( "decoder_cell1", self.config["Decoder"]["cell_type"], int(self.config["Decoder"]["hidden_size"]), 1,dropout_param=self.dropout_param)[0])
        
            var_name = "new_decoder"
            self.decoder_cell_second = []
            with tf.variable_scope("Decoder_Copy_Cell_New", reuse=tf.AUTO_REUSE):
                self.decoder_cell_second.append(self.model.model.select_cell( "decoder_cell", self.config["Decoder"]["cell_type"], int(self.config["Decoder"]["hidden_size"]), 1,  dropout_param=self.dropout_param)[0])

            with tf.variable_scope("Decoder_Copy_Cell_Second_New", reuse=tf.AUTO_REUSE):
                self.decoder_cell_second.append(self.model.model.select_cell( "decoder_cell1", self.config["Decoder"]["cell_type"], int(self.config["Decoder"]["hidden_size"]), 1, dropout_param=self.dropout_param)[0])
            

            beam_size = int(self.config["BeamSearch"]["beam_size"])
            batch_size = int(self.config["BeamSearch"]["batch_size"])

            self.model.model.is_train = False
            self.model.pre_pipeline(self.encoder_input_batch, self.query_input_batch, self.decoder_input_batch, self.feed_previous_placeholder,self.feed_true_placeholder)
            print ("INIITIAL STATES---------------", self.model.initial_states)

            self.memory_states, self.prev_time_step_data, self.query_state,self.masked_weights,self.content_weights, self.prev_decoder_tokens, self.prev_attention_weights = self.add_placeholders_beamsearch(self.encoder_input_batch,self.model.initial_states)

            print ("The beam search memory states, query state")
            print("---"*30, self.memory_states.get_shape(), self.query_state)
            for i in self.prev_time_step_data:
                print(i, self.prev_time_step_data[i])

            # encoder_input_batch =
            # def decode_one_step(self, memory_states, prev_time_step_data, query_state,word_embeddings,decoder_cell, content_weights, content_prop_weights, word_indices, prop_indices, is_inference=False, var_name="", is_prev_decoder=True):

            self.new_state_values = self.model.model.decode_one_step_first_decoder(self.encoder_input_batch,self.memory_states, self.prev_time_step_data, self.query_state, self.model.model.word_embeddings, self.decoder_cell, content_weights=self.content_weights,content_prop_weights=None,word_indices=self.prev_time_step_data["words"],prop_indices= None,is_inference=True,var_name="", is_prev_decoder=True)
            print ("<=>="*30)

            self.initial_states_2nd = self.model.model._init_decoder_params(self.model.model.memory_states.get_shape()[-1].value, is_prev_state=None, var_name="new_decoder")

            self.query_state_batch = tf.placeholder(tf.float32, shape=(batch_size, 2*int(self.config["Query"]["hidden_size"])), name="query_state_batch_pl")
            self.prev_decoder_states, self.masked_weights_props, self.prev_decoder_mean_attn, self.total_encoder_teacher_attention_weights, self.question_gate, _ = self.model.post_first_decoder(self.prev_decoder_tokens, tf.unstack(self.prev_attention_weights,axis=1), self.query_state_batch)
            

            self.prev_time_step_data_second, self.masked_weights_props_pl, self.prev_decoder_mean_attn_pl, self.question_gate_pl = self.add_placeholders_beamsearch_second(self.encoder_input_batch,self.model.initial_states)

            self.new_state_values_second = self.model.model.decode_one_step_second_decoder(self.encoder_input_batch, self.memory_states,
                                          self.prev_time_step_data_second, self.query_state, self.model.model.word_embeddings, self.decoder_cell_second,
                                          content_weights=self.content_weights,content_prop_weights=None,word_indices=self.prev_time_step_data_second["words"],
                                          prop_indices= None,is_inference=True,var_name="new_decoder", is_prev_decoder=False,
                                          masked_weights_props = self.masked_weights_props_pl , prev_decoder_mean_attn=self.prev_decoder_mean_attn_pl, question_gate=self.question_gate_pl)
                                          #masked_weights_props = self.masked_weights_props_pl , prev_decoder_mean_attn=self.prev_decoder_mean_attn_pl, question_gate=self.question_gate_pl)
            print ("<=>="*30)
            # self.new_state_values = self.model.model.decode_one_step
          
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()
            self.saver = saver
            #c = tf.ConfigProto()
            # Create a session for running Ops on the Graph.
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            sess = tf.Session(config=config)


            # Instantiate a SummaryWriter to output summaries and the Graph.


            # if best_model exists pick the weights from there:
            if (os.path.exists(os.path.join(self.config["Log"]["output_dir"] , "best_model.meta"))):
                print ("Initializing the model with the best saved model")
                optimistic_restore(sess, os.path.join(self.config["Log"]["output_dir"] , "best_model"))

                saver.save(sess, os.path.join(self.config["Log"]["output_dir"], "saved_params"))
                #self.print_titles_in_files(sess, self.dataset.datasets["test"])

                #self.print_titles_in_files_beamsearch(sess, self.dataset.datasets["test"], self.dataset.vocab, 101)
                print ("wrote the file")

            print('Trainable Variables') 
            print ('\n'.join([v.name for v in tf.trainable_variables()]))
            best_val_loss = float('inf')

            # To store the model that gives the best result on validation.
            best_val_epoch = 0

            self.print_titles_in_files_beamsearch(sess,self.dataset.datasets["test"], self.dataset.vocab)
            #self.print_titles_in_files_beamsearch(sess, self.dataset.datasets["valid"], self.dataset.vocab)
            # print ("Test Loss:{}".format(test_loss))



    def print_titles_in_files_beamsearch(self, sess, data_set, vocab, epoch = 1000):

        """ Prints the titles for the requested examples.

        Args:
              sess: Running session of tensorflow
              data_set : Dataset from which samples will be retrieved.
              total_examples: Number of samples for which title is printed.

        """
        # self.config["Hyperparams"]["batch_size"] = str(1)
        total_loss = 0
        beam_size = int(self.config["BeamSearch"]["beam_size"])
        f1 = codecs.open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_beam_"+str(beam_size)+"final_results_first"+str(epoch)), "w", encoding="utf-8")
        f2 = codecs.open(os.path.join(self.config["Log"]["output_dir"], data_set.name+ "_beam" + str(beam_size) + "_attention_weights_first"+ str(epoch)), "w", encoding="utf-8")
        #f3 = codecs.open(os.path.join(self.config["Log"]["output_dir"], data_set.name+ "_beam_output_switches"+str(epoch)), "w", encoding="utf-8")
        f3 = codecs.open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_beam_"+str(beam_size)+"final_results_second"+str(epoch)), "w", encoding="utf-8")
        f4 = codecs.open(os.path.join(self.config["Log"]["output_dir"], data_set.name+ "_beam" + str(beam_size) + "_attention_weights_second"+ str(epoch)), "w", encoding="utf-8")


        steps_per_epoch =  int(math.ceil(float(data_set.number_of_examples) / int(self.config["BeamSearch"]["batch_size"])))
        
        batch_size = int(self.config["BeamSearch"]["batch_size"])

       
        for step in xrange(steps_per_epoch):
            encoder_inputs, query_inputs, decoder_inputs = self.dataset.next_batch(data_set,batch_size, False)


            hyps, attn_weights, tokens_predicted, encoder_outputs, encoder_state, query_outputs, query_state, attention_values = self.run_beam_search(sess, vocab, self.encoder_input_batch, encoder_inputs,self.query_input_batch, query_inputs)
            #shapes = [i.shape for i in attention_values[0]]
            #print (shapes)
            #attention_values = [np.sum(np.stack(i, axis=1), axis=1) for i in attention_values]
            
            print(attention_values)
            hyps_second,attn_weights_second = self.run_beam_search_2nd( sess, vocab, self.encoder_input_batch, encoder_inputs, 
                         self.query_input_batch, query_inputs, tokens_predicted, encoder_outputs, encoder_state=encoder_state, query_outputs=query_outputs, query_state=query_state, 
                         decoder_inputs = decoder_inputs, prev_attention_values=attention_values)
            #hyps = decoded_sentences
            # Have to encode the final tokens
            # final_tokens = [i.tokens for i in hyps]
            #hyps_second, attn_weights_second = self.run_beam_search_second_decoder(sess, self.encoder_input_batch, encoder_inputs, self.query_input_batch, query_inputs, 
            #                                                                        encoder_outputs, encoder_state, query_state, final_tokens)

            sys.stdout.flush()
         
            for i in hyps_second:
                f3.write(i + "\n")
            for j in attn_weights_second:
                f4.write(" ".join([str(m) for m in j]) + "\n")

         
            for i in hyps:
                f1.write(i + "\n")
            for j in attn_weights:
                f2.write(" ".join([str(m) for m in j]) + "\n")
            continue

            print ("values", hyps)
            _decoder_states_ = [i.tokens for i in hyps]
            attention_weights = [i.attn_dists for i in hyps]
            #print('attn_wt shape', np.shape(attention_weights))
            #print('dec_state shape', np.shape(_decoder_states_))
          

            attention_states = np.reshape(np.array([np.argmax(i) for i in attention_weights]), (-1, 1))
            # Pack the list of size max_sequence_length to a tensor
            decoder_states = np.reshape(np.array([i for i in _decoder_states_]), (-1, 1))
            #output_switches = np.reshape(output_switches,(64, len(output_switches)))
            # tensor will be converted to [batch_size * sequence_length * symbols]
            ds = np.transpose(decoder_states)
            attention_states = np.transpose(attention_states)
            attn_state = attention_states 
            # Converts this to a length of batch sizes
            final_ds = ds.tolist()
            final_as = attn_state.tolist()
            #output_switches = output_switches.tolist()
            print(len(final_ds))
            for i, states in enumerate(final_ds):
                # Get the index of the highest scoring symbol for each time step
                #indexes = sess.run(tf.argmax(states, 1))
                print (states)
                s =  self.dataset.decode_to_sentence(states)
                print(s)
                f1.write(s + "\n")
                x = " ".join(str(m) for m in final_as[i])
                #y = " ".join(str(m) for m in output_switches[i])
                f2.write(x + "\n")

            sys.stdout.flush()
        return
    
    def add_placeholders_beamsearch(self,encoder_input_batch ,initial_states_place):

        

        batch_size = int(self.config["BeamSearch"]["batch_size"])
        beam_size = int(self.config["BeamSearch"]["beam_size"])
        dec_hidden_size = 2* int(self.config["Encoder"]["hidden_size"])

        query_state = tf.zeros(shape=(batch_size * beam_size, 2*int(self.config["Query"]["hidden_size"])), dtype=tf.float32)
        
        prev_time_step_data = {}

        prev_time_step_data["content_weights"]  =  tf.reshape(tf.tile(tf.expand_dims(encoder_input_batch["content_weights"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size,encoder_input_batch["content_weights"].shape[-1]))
        prev_time_step_data["words"]  =   tf.reshape(tf.tile(tf.expand_dims(encoder_input_batch["word"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, encoder_input_batch["word"].shape[-1] ))
        prev_time_step_data["token"] = tf.placeholder(tf.int32, shape=(batch_size*beam_size,), name="token_1")
        prev_time_step_data["attns_state"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["attns_state"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size,initial_states_place["attns_state"].shape[-1]))      
        
        prev_time_step_data["state_h"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_h"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_h"].shape[-1] ))      
        prev_time_step_data["state_c"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_c"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_c"].shape[-1] )) 

        prev_time_step_data["state_temp_h"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_h"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_h"].shape[-1] ))      
        prev_time_step_data["state_temp_c"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_c"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_c"].shape[-1] )) 

        # prev_time_step_data["property_state"] = tf.reshape(tf.tile(tf.expand_dims(initial_states_place["property_state"], axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, 2*int(self.config["Encoder"]["hidden_size"])))
        
        memory_states = tf.zeros(shape=(batch_size*beam_size, self.model.model.memory_states.get_shape()[1].value, dec_hidden_size), dtype=tf.float32) 

        # word_indices = tf.placeholder(shape=(batch_size*beam_size, int(self.config["Encoder"]["max_sequence_length"] ))
        content_weights = tf.placeholder(shape=(batch_size*beam_size,int(self.config["Encoder"]["max_sequence_length"])),dtype=tf.float32)

        prev_time_step_data["prev_coverage"]  = tf.placeholder(tf.float32, shape=(batch_size*beam_size, int(self.config["Encoder"]["max_sequence_length"])), name="prev_coverage_placeholder")
        prev_time_step_data["prev_coverage_between_decoders"]  = tf.placeholder(tf.float32, shape=(batch_size*beam_size, int(self.config["Decoder"]["max_prop_steps"])-1), name="between_decoder_placeholder")

        masked_weights = tf.placeholder(tf.int32, shape=(batch_size * beam_size,) , name="masked_weights")

        print ("In add placeholder mem states shape query {} state shape{}".format(memory_states,query_state))
        prev_decoder_tokens = tf.placeholder(tf.int32, shape=(batch_size,int(self.config["Decoder"]["max_sequence_length"])-1) , name="prev_decoder_tokens")
        prev_attention_weights = tf.placeholder(tf.float32, shape=(batch_size,int(self.config["Decoder"]["max_sequence_length"])-1, int(self.config["Encoder"]["max_sequence_length"]) ), name="prev_attention_weights")


        return memory_states,prev_time_step_data,query_state, masked_weights, content_weights, prev_decoder_tokens, prev_attention_weights

    def add_placeholders_beamsearch_second(self,encoder_input_batch ,initial_states_place):

        batch_size = int(self.config["BeamSearch"]["batch_size"])
        beam_size = int(self.config["BeamSearch"]["beam_size"])
        dec_hidden_size = 2* int(self.config["Encoder"]["hidden_size"])
        query_state = tf.placeholder(tf.float32, shape=(batch_size * beam_size, 2*int(self.config["Query"]["hidden_size"])), name="query__")
        
        prev_time_step_data = {}

        prev_time_step_data["content_weights"]  =  tf.reshape(tf.tile(tf.expand_dims(encoder_input_batch["content_weights"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size,encoder_input_batch["content_weights"].shape[-1]))
        prev_time_step_data["words"]  =   tf.reshape(tf.tile(tf.expand_dims(encoder_input_batch["word"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, encoder_input_batch["word"].shape[-1] ))
        prev_time_step_data["token"] = tf.placeholder(tf.int32, shape=(batch_size*beam_size,), name="token_second")
        prev_time_step_data["attns_state"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["attns_state"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size,initial_states_place["attns_state"].shape[-1]))      
        prev_time_step_data["combined_attns_state"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["combined_attns_state"],axis=1), 
                                                       multiples=[1,beam_size, 1]), (batch_size*beam_size,initial_states_place["combined_attns_state"].shape[-1])) 
        attns_state_different = tf.placeholder(dtype=tf.float32, shape=(batch_size*beam_size, 2*int(self.config["Encoder"]["hidden_size"]) ), name="attns_state_different")

        if self.config["Question_encoder"]["use_question_encoder"] == "True" and self.config["Question_encoder"]["word_concat"] == "True":
            new_size = 2*int(self.config["Question_encoder"]["hidden_size"]) + int(self.config["Embedding"]["word_embedding_size"])
            prev_decoder_attn_state = tf.zeros((batch_size * beam_size , new_size), dtype=tf.float32)
            prev_decoder_states = tf.placeholder(tf.float32, shape=(batch_size*beam_size, int(self.config["Decoder"]["max_sequence_length"])-1, new_size ), name="prev_decoder_state")
        elif self.config["Question_encoder"]["use_question_encoder"] == "True": 
            prev_decoder_attn_state = tf.zeros((batch_size*beam_size, 2*int(self.config["Question_encoder"]["hidden_size"])), dtype=tf.float32)
            prev_decoder_states = tf.placeholder(tf.float32, shape=(batch_size*beam_size, int(self.config["Decoder"]["max_sequence_length"])-1, 2*int(self.config["Question_encoder"]["hidden_size"]) ), name="prev_decoder_state")
        ####   Initial States when Question_encoder = False
        else:
            prev_decoder_attn_state = tf.zeros((batch_size*beam_size, int(self.config["Embedding"]["word_embedding_size"]) ), dtype=tf.float32)   
            prev_decoder_states = tf.placeholder(tf.float32, shape=(batch_size*beam_size, int(self.config["Decoder"]["max_sequence_length"])-1, int(self.config["Embedding"]["word_embedding_size"])), name="prev_decoder_state")

        prev_time_step_data["prev_decoder_attn_state"] =  prev_decoder_attn_state
        prev_time_step_data["attns_state_different"] = attns_state_different
        prev_time_step_data["prev_decoder_states"] = prev_decoder_states

        prev_time_step_data["state_h"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_h"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_h"].shape[-1] ))      
        prev_time_step_data["state_c"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_c"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_c"].shape[-1] )) 

        prev_time_step_data["state_temp_h"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_h"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_h"].shape[-1] ))      
        prev_time_step_data["state_temp_c"] =  tf.reshape(tf.tile(tf.expand_dims(initial_states_place["decoder_state_c"],axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, initial_states_place["decoder_state_c"].shape[-1] )) 
 
        # prev_time_step_data["property_state"] = tf.reshape(tf.tile(tf.expand_dims(initial_states_place["property_state"], axis=1), multiples=[1,beam_size, 1]), (batch_size*beam_size, 2*int(self.config["Encoder"]["hidden_size"])))
        
        memory_states = tf.zeros(shape=(batch_size*beam_size, self.model.model.memory_states.get_shape()[1].value, dec_hidden_size), dtype=tf.float32) 


        # word_indices = tf.placeholder(shape=(batch_size*beam_size, int(self.config["Encoder"]["max_sequence_length"] ))
        content_weights = tf.placeholder(shape=(batch_size*beam_size,int(self.config["Encoder"]["max_sequence_length"])),dtype=tf.float32)

        prev_time_step_data["prev_coverage"]  = tf.placeholder(tf.float32, shape=(batch_size*beam_size, int(self.config["Encoder"]["max_sequence_length"])), name="prev_coverage")
        prev_time_step_data["prev_coverage_vec_different"] = tf.placeholder(tf.float32, (batch_size*beam_size,int(self.config["Encoder"]["max_sequence_length"])), name="prev_coverage_vec_different")
        prev_time_step_data["prev_coverage_between_decoders"]  = tf.placeholder(tf.float32, (batch_size*beam_size, int(self.config["Decoder"]["max_prop_steps"])-1), name="between_decoder")
        #prev_time_step_data["prev_coverage_vec_different"] = tf.zeros((batch_size*beam_size, int(self.config["Decoder"]["max_prop_steps"]) -1))

        
        masked_weights = tf.placeholder(tf.int32, shape=(batch_size * beam_size,) , name="masked_weights_pl")

        masked_weights_props = tf.placeholder(tf.int32, shape=(batch_size * beam_size,int(self.config["Decoder"]["max_sequence_length"])-1) , name="masked_weights")
        prev_decoder_mean_attn = tf.placeholder(tf.float32, shape=(batch_size * beam_size,int(self.config["Encoder"]["max_sequence_length"])) , name="prev_Decoder_mean")
        question_gate = tf.placeholder(tf.float32, shape=(batch_size * beam_size,), name="question_gate")


        print ("In Second add placeholder mem states shape query {} state shape{}".format(memory_states,query_state))

        return prev_time_step_data, masked_weights_props, prev_decoder_mean_attn, question_gate

    def run_beam_search(self, sess, vocab, encoder_input_batch, encoder_inputs, query_input_batch, query_inputs):
        """Performs beam search decoding on the given example.
        Args:
          sess: a tf.Session
          model: a seq2seq model
          vocab: Vocabulary object
          batch: Batch object that is the same example repeated across the batch
        Returns:
          best_hyp: Hypothesis object; the best hypothesis found by beam search.
        """
        # Run the encoder to get the encoder hidden states and decoder initial state

        self.model.dropout_param = encoder_input_batch["dropout"]
        max_steps = int(self.config["Decoder"]["max_decoder_steps"]) - 1 
        beam_size = int(self.config["BeamSearch"]["beam_size"])
        batch_size = int(self.config["Hyperparams"]["batch_size"])
        max_encoder_length = int(self.config["Encoder"]["max_sequence_length"])
        dec_hidden_size = 2 * int(self.config["Encoder"]["hidden_size"])
        batch_size = int(self.config["BeamSearch"]["batch_size"])
        max_length = int(self.config["Encoder"]["max_sequence_length"])

        prev_coverage_bool = self.config["Decoder"]["use_coverage"] == "True"
        include_prop_weights = self.config["Decoder"]["include_prop_weights"] == "True"

        results = [list([]) for _ in range(batch_size)]
        steps = 0
        prev_time_step_data_val = {}    

        feed_dict = {
        encoder_input_batch["word"]: encoder_inputs["word"],
        encoder_input_batch["char"]: encoder_inputs["char"],
        encoder_input_batch["seq_length"]: encoder_inputs["seq_length"],
        encoder_input_batch["dropout"]: 1.0,
        encoder_input_batch["positional"] : encoder_inputs["positional"],
        query_input_batch["word"] : query_inputs["word"],
        query_input_batch["char"] : query_inputs["char"],
        query_input_batch["seq_length"]: query_inputs["seq_length"],
        encoder_input_batch["content_weights"]: encoder_inputs["content_weights"],
        query_input_batch["position"]: query_inputs["position"],
        self.feed_previous_placeholder: True,
        self.encode_sequence_length : encoder_inputs["seq_length"],
        }

        # get the initial states and encoder outputs
        encoder_outputs, encoder_state, query_outputs, query_state, initial_states = sess.run([self.model.encoder_outputs, self.model.encoder_state, self.model.query_outputs, self.model.query_state,self.model.initial_states], feed_dict=feed_dict)


        dec_in_state_h = initial_states['decoder_state_h']
        dec_in_state_c = initial_states['decoder_state_c']
        initial_states_place = self.model.initial_states


        # property_states = np.mean(encoder_outputs[:, max_length:,:], axis=1)
        hyps_batch =  [ [ Hypothesis(tokens=[vocab.encode_word("<s>")],
                           probs=[1.0],
                           state_h=dec_in_state_h[i],
                           state_c=dec_in_state_c[i],
                           state_temp_h = dec_in_state_h[i],
                           state_temp_c = dec_in_state_c[i],
                           prev_coverage_vec = initial_states["prev_coverage"][i],
                           attn_state = initial_states['attns_state'][i],
                           attn_values = [],
                           ) for  _ in xrange(int(self.config["BeamSearch"]["beam_size"]))] for i in xrange(batch_size) ]  


        #encoder_inputs_temp = copy.deepcopy(encoder_inputs)
        for key in encoder_inputs:
            encoder_inputs[key] = np.repeat(encoder_inputs[key],beam_size,axis=0)

        prev_time_step_data_val = {}
        memory_states = np.repeat(encoder_outputs,beam_size,axis=0)
        query_states = np.repeat(query_state,beam_size,axis=0)
                
        content_weights = encoder_inputs["content_weights"]
        words = encoder_inputs["word"]

        while steps < int(self.config["Decoder"]["max_decoder_steps"]):
          latest_tokens = [h.latest_token for hyps in hyps_batch for h in hyps ] 
          states_h =  [h.state_h for hyps in hyps_batch for h in hyps ] # list of current decoder states of the hypotheses
          states_c = [h.state_c for hyps in hyps_batch for h in hyps] # list of current decoder states of the hypotheses
          states_temp_h =  [h.state_temp_h for hyps in hyps_batch for h in hyps ] # list of current decoder states of the hypotheses
          states_temp_c = [h.state_temp_c for hyps in hyps_batch for h in hyps] # list of current decoder states of the hypotheses
          attns_state = [h.attn_state for hyps in hyps_batch for h in hyps]
          prev_coverage = [h.prev_coverage_vec for hyps in hyps_batch for h in hyps]
        #   prop_state = [h.property_state for hyps in hyps_batch for h in hyps]

          prev_time_step_data_val["token"] = np.asarray(latest_tokens)#.reshape(batch_size*beam_size,-1)
          prev_time_step_data_val["attns_state"] = np.asarray(attns_state)#.reshape(batch_size*beam_size,-1)
          prev_time_step_data_val["content_weights"] = np.asarray(content_weights)#.reshape(batch_size*beam_size,-1)
          prev_time_step_data_val["words"] = np.asarray(words)#.reshape(batch_size*beam_size,-1)
          prev_time_step_data_val["state_h"] = np.asarray(states_h)#.reshape(batch_size*beam_size,-1)
          prev_time_step_data_val["state_temp_h"] = np.asarray(states_temp_h)#.reshape(batch_size*beam_size,-1)
          prev_time_step_data_val["state_c"] = np.asarray(states_c)#.reshape(batch_size*beam_size,-1)
          prev_time_step_data_val["state_temp_c"] = np.asarray(states_temp_c)#.reshape(batch_size*beam_size,-1)
        #   prev_time_step_data_val["property_state"] = np.asarray(prop_state)
          prev_time_step_data_val["prev_coverage_vec"] = np.asarray(prev_coverage)

          prev_tokens = self.prev_time_step_data["token"]
          prev_state_h   = self.prev_time_step_data["state_h"]
          prev_state_c   = self.prev_time_step_data["state_c"]
          prev_state_temp_h = self.prev_time_step_data["state_temp_h"]
          prev_state_temp_c = self.prev_time_step_data["state_temp_c"]
          prev_attns_state = self.prev_time_step_data["attns_state"]
        #   prev_prop_state = self.prev_time_step_data.get("property_state")
          prev_coverage_vec = self.prev_time_step_data["prev_coverage"]



          feed_dict = {

          prev_tokens : prev_time_step_data_val["token"],
          prev_state_h : prev_time_step_data_val["state_h"],
          prev_state_c : prev_time_step_data_val["state_c"],
          prev_state_temp_h : prev_time_step_data_val["state_temp_h"],
          prev_state_temp_c : prev_time_step_data_val["state_temp_c"],
          self.prev_time_step_data["words"] : words,
          prev_attns_state : prev_time_step_data_val["attns_state"],
          self.prev_time_step_data["content_weights"] : content_weights, 
          prev_coverage_vec :prev_time_step_data_val["prev_coverage_vec"],
          self.memory_states : memory_states,
          self.masked_weights : encoder_inputs["seq_length"],
          #self.query_state : query_states ,
          self.feed_previous_placeholder: True,
          encoder_input_batch["dropout"]: 1.0,
          encoder_input_batch["seq_length"]: encoder_inputs["seq_length"],
          self.content_weights: content_weights,
          
          }
          if self.config["Query"]["use_query"] == "True":
              feed_dict[self.query_state] = query_states 

          new_time_step_data = {}
          new_time_step_data = sess.run(self.new_state_values, feed_dict=feed_dict)

          topk_ids = np.flip(new_time_step_data["comb_projection"].argsort(axis=1)[:,-2*beam_size:], 1)
          probs_temp = copy.deepcopy(new_time_step_data["comb_projection"])
          probs_temp.sort(axis=1)
          probs = np.flip(probs_temp[:, -2*beam_size:], 1)

          states_h_new = new_time_step_data["state"].h
          states_c_new = new_time_step_data["state"].c
          states_temp_h_new = new_time_step_data["state_temp"].h
          states_temp_c_new = new_time_step_data["state_temp"].c
          attns_states_new = new_time_step_data["attns_state"]
          attns_values = new_time_step_data["attention_values"][0]
          prev_coverages = new_time_step_data["prev_coverage"]
        #   property_states = new_time_step_data["property_state"]

          all_hyps = []
          # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
          for batch in xrange(batch_size):
              num_orig_hyps = 1 if steps == 0 else len(hyps_batch[batch])
              all_hyps = []
              for i in xrange(num_orig_hyps):
                  new_state_h, new_state_c=  states_h_new[batch*beam_size + i],states_c_new[batch*beam_size + i]
                  new_state_temp_h, new_state_temp_c, new_prev_coverage = states_temp_h_new[batch*beam_size + i], states_temp_c_new[batch*beam_size + i], prev_coverages[batch*beam_size + i]
                  h = hyps_batch[batch][i]
                  for j in xrange(int(self.config["BeamSearch"]["beam_size"]) * 2):  # for each of the top 2*beam_size hyps:
                        # Extend the ith hypothesis with the jth option
                        new_hyp = h.extend(token=topk_ids[batch*beam_size + i, j],
                                     prob=probs[batch*beam_size + i, j],
                                     state_h=new_state_h,
                                     state_c = new_state_c,
                                     state_temp_h=new_state_temp_h, 
                                     state_temp_c=new_state_temp_c,
                                     prev_coverage_vec = new_prev_coverage,
                                     attn_values=attns_values[batch*beam_size + i],
                                     attn_state = attns_states_new[batch*beam_size + i])

                        all_hyps.append(new_hyp)
              #assert(np.all(new_state != states[i]))

              # Filter and collect any hypotheses that have produced the end token.
              hyps = [] # will contain hypotheses for the next step
              for h in sort_hyps(all_hyps): # in order of most likely h
                if h.latest_token == vocab.word_to_index["<eos>"]: # if stop token is reached...
                     # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                    if steps >= int(self.config["Decoder"]["min_dec_steps"]):
                        results[batch].append(h)
                else: # hasn't reached stop token, so continue to extend this hypothesis
                    hyps.append(h)
                if len(hyps) == int(self.config["BeamSearch"]["beam_size"]) : #or len(results[batch]) == int(self.config["BeamSearch"]["beam_size"]):
                    # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                    break
              #hyps_batch.append(hyps)
              hyps_batch[batch]= hyps
          steps += 1

        # At this point, either we've got beam_size results, or we've reached maximum decoder steps
        for b in range(batch_size):
            if len(results[b])==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
                  results[b] = hyps_batch[b]

        # Sort hypotheses by average log probability
        top_sentences = []
        attention_dists = []
        attention_values = []
        tokens_predicted = []

        for batch in xrange(batch_size):
           print ("Batch Number", batch)
           hyps_sorted = sort_hyps(results[batch])
           tokens_predicted.append(hyps_sorted[0].tokens)
           top_sentences.append(hyps_sorted[0])

        tokens_predicted = [i[1:] for i in tokens_predicted]
        token_predicted,_ = self.dataset.pad_data(tokens_predicted, max_steps)
        print (token_predicted)
        decoded_sentences = []
        for i in top_sentences:
           temp = self.dataset.decode_to_sentence(i.tokens[1:])
           print ("Decoder1 ", temp)
           decoded_sentences.append(temp)
           attn_dists = i.attn_dists
           correct_tokens = [np.argmax(k) for k in attn_dists]
           attention_dists.append(correct_tokens)
           

           if len(attn_dists) <= int(self.config["Decoder"]["max_sequence_length"])-1:           
                attn_dists = attn_dists + [[0]*max_encoder_length]*(max_steps - len(attn_dists))
           else:
                attn_dists = attn_dists[:max_steps]

           attn_dists = np.asarray(attn_dists)
           attention_values.append(attn_dists)

        return decoded_sentences, attention_dists, np.asarray(token_predicted), encoder_outputs, encoder_state, query_outputs, query_state, attention_values



    def run_beam_search_2nd(self, sess, vocab, encoder_input_batch,
                                      encoder_inputs, query_input_batch, query_inputs,
                                      tokens_predicted, encoder_outputs, encoder_state, query_outputs, 
                                      query_state, decoder_inputs, prev_attention_values):



        # init variables
        self.model.dropout_param = encoder_input_batch["dropout"]
        max_steps = int(self.config["Decoder"]["max_sequence_length"]) - 1
        beam_size = int(self.config["BeamSearch"]["beam_size"])
        batch_size = int(self.config["BeamSearch"]["batch_size"])
        max_encoder_length = int(self.config["Encoder"]["max_sequence_length"])
        dec_hidden_size = 2 * int(self.config["Encoder"]["hidden_size"])
        prev_coverage_bool = self.config["Decoder"]["use_coverage"] == "True"
        include_prop_weights = self.config["Decoder"]["include_prop_weights"] == "True"


        results = [list([]) for _ in range(batch_size)]
        steps = 0
        prev_time_step_data_val = {}
        #We have the encoder states and outputs already, post first decoder needs to be run next.
        #stack attention values to size batch-size x timesteps x sequence_length
        prev_attention_values = np.stack(prev_attention_values, axis=0)

        print ("prev_attention values", prev_attention_values.shape)
        feed_dict = {
            self.prev_decoder_tokens : tokens_predicted,
            self.model.dropout_param : 1.0,
            self.prev_attention_weights: prev_attention_values,
            self.query_state_batch : query_state
        }

        prev_decoder_states, masked_weights_props, prev_decoder_mean_attn, prev_coverage_vec_different, question_gate = sess.run(
            [self.prev_decoder_states, self.masked_weights_props, self.prev_decoder_mean_attn, self.total_encoder_teacher_attention_weights, 
            self.question_gate], feed_dict = feed_dict)

        # get the inital states 
        feed_dict = {
            self.model.dropout_param : 1.0,
            self.model.query_state : query_state,
            self.model.encoder_state : encoder_state,
        }
        initial_states = sess.run(self.initial_states_2nd, feed_dict=feed_dict)

        # get the initial variables
        attns_state = initial_states["attns_state"]
        combined_attns_state = initial_states["combined_attns_state"]
        prev_decoder_attn_state = initial_states["prev_decoder_attn_state"]
        attns_state_different = initial_states["attns_state_different"]
        dec_in_state_h = initial_states["decoder_state_h"]
        dec_in_state_c = initial_states["decoder_state_c"]
        prev_coverage_between_decoders = initial_states["prev_coverage_between_decoders"]
        prev_coverage = initial_states["prev_coverage"]
        content_weights = encoder_inputs["content_weights"]
        words = encoder_inputs["word"]

        # Got all the ingredients for 2nd Decoder. Initializing the hypothesis now.

        # only store the variables that change during each timestep
        hyps_batch  = [[ Hypothesis_2nd(tokens = [vocab.encode_word("<s>")],
                         probs = [1.0],
                         state_h = dec_in_state_h[i],
                         state_c = dec_in_state_c[i],
                         state_temp_h = dec_in_state_h[i],
                         state_temp_c = dec_in_state_c[i],
                         prev_coverage_vec = prev_coverage[i],
                         prev_coverage_between_decoders = prev_coverage_between_decoders[i],
                         prev_coverage_vec_different = prev_coverage_vec_different[i],
                         attns_state_different = attns_state_different[i],
                         combined_attn_state = combined_attns_state[i],
                         attn_values = [],
        ) for _ in xrange(beam_size) ] for i in xrange(batch_size) ]
                     
        
        def repeat_beam(x, beam_size):
            return np.repeat(x, beam_size, axis=0)


        
        prev_decoder_states = repeat_beam(prev_decoder_states, beam_size)
        masked_weights_props = repeat_beam(masked_weights_props, beam_size)
        prev_decoder_mean_attn = repeat_beam(prev_decoder_mean_attn, beam_size)
        prev_coverage_vec_different = repeat_beam(prev_coverage_vec_different, beam_size)
        question_gate = repeat_beam(question_gate, beam_size)
        memory_states = repeat_beam(encoder_outputs, beam_size)
        query_states  = repeat_beam(query_state, beam_size)
        
        print ("III", prev_decoder_mean_attn) 
        
        #start the loop over the decoder steps
        while steps < int(self.config["Decoder"]["max_decoder_steps"]):
            latest_tokens = [h.latest_token for hyps in hyps_batch for h in hyps ] 
            states_h =  [h.state_h for hyps in hyps_batch for h in hyps ] # list of current decoder states of the hypotheses
            states_c = [h.state_c for hyps in hyps_batch for h in hyps] # list of current decoder states of the hypotheses
            states_temp_h =  [h.state_temp_h for hyps in hyps_batch for h in hyps ] # list of current decoder states of the hypotheses
            states_temp_c = [h.state_temp_c for hyps in hyps_batch for h in hyps] # list of current decoder states of the hypotheses
            prev_coverage = [h.prev_coverage_vec for hyps in hyps_batch for h in hyps]
            prev_coverage_between_decoders = [h.prev_coverage_between_decoders for hyps in hyps_batch for h in hyps]
            prev_coverage_vec_different = [h.prev_coverage_vec_different for hyps in hyps_batch for h in hyps]
            combined_attns_state = [h.combined_attn_state for hyps in hyps_batch for h in hyps]
            attns_state_different = [h.attns_state_different for hyps in hyps_batch for h in hyps]


            prev_tokens = self.prev_time_step_data["token"]
            prev_state_h   = self.prev_time_step_data["state_h"]
            prev_state_c   = self.prev_time_step_data["state_c"]
            prev_state_temp_h = self.prev_time_step_data["state_temp_h"]
            prev_state_temp_c = self.prev_time_step_data["state_temp_c"]
            prev_attns_state = self.prev_time_step_data["attns_state"]
            prev_coverage_vec = self.prev_time_step_data["prev_coverage"]


            # Create feed dictionary
            feed_dict = {
                self.prev_time_step_data_second["token"] : np.asarray(latest_tokens),
                self.prev_time_step_data_second["state_h"] : np.asarray(states_h),
                self.prev_time_step_data_second["state_c"] : np.asarray(states_c),
                self.prev_time_step_data_second["state_temp_h"] : np.asarray(states_temp_h),
                self.prev_time_step_data_second["state_temp_c"] : np.asarray(states_temp_c),
                self.prev_time_step_data_second["prev_coverage"] : np.asarray(prev_coverage),
                self.prev_time_step_data_second["prev_coverage_between_decoders"]: np.array(prev_coverage_between_decoders),
                self.prev_time_step_data_second["prev_coverage_vec_different"] : np.asarray(prev_coverage_vec_different),
                self.prev_time_step_data_second["combined_attns_state"]: np.asarray(combined_attns_state),
                self.prev_time_step_data_second["attns_state_different"] : np.asarray(attns_state_different),
                self.prev_time_step_data_second["prev_decoder_states"] : prev_decoder_states,

                self.prev_time_step_data_second["words"] : words,
                self.prev_time_step_data_second["content_weights"] : content_weights,
                self.memory_states : memory_states,
                self.masked_weights : encoder_inputs["seq_length"],
                self.model.dropout_param : 1.0,
                encoder_input_batch["seq_length"]: encoder_inputs["seq_length"],
                self.content_weights : content_weights,
                self.masked_weights_props_pl : masked_weights_props,
                self.prev_decoder_mean_attn_pl : prev_decoder_mean_attn,
                self.question_gate_pl : np.asarray(question_gate)
            }

            if self.config["Query"]["use_query"] == "True":
                    feed_dict[self.query_state] = query_states 
	
            #print (self.new_state_values_second)
            print (np.asarray(prev_decoder_mean_attn).shape, np.asarray(question_gate).shape)
            new_time_step_data = sess.run(self.new_state_values_second, feed_dict= feed_dict)

            topk_ids = np.flip(new_time_step_data["comb_projection"].argsort(axis=1)[:,-2*beam_size:], 1)
            probs_temp = copy.deepcopy(new_time_step_data["comb_projection"])
            probs_temp.sort(axis=1)
            probs = np.flip(probs_temp[:, -2*beam_size:], 1)

            # get the new values
            states_h_new = new_time_step_data["state"].h
            states_c_new = new_time_step_data["state"].c
            states_temp_h_new = new_time_step_data["state_temp"].h
            states_temp_c_new = new_time_step_data["state_temp"].c
            combined_attns_states_new = new_time_step_data["combined_attns_state"]
            attns_state_different_new = new_time_step_data.get("attns_state_different")
            attns_values = new_time_step_data["total_attention"][0]
            prev_coverages_new = new_time_step_data["prev_coverage"]
            prev_coverage_between_decoders_new = new_time_step_data["prev_coverage_between_decoders"]
            prev_coverage_vec_different_new = new_time_step_data["prev_coverage_vec_different"]

            all_hyps = [] 
            for batch in xrange(batch_size):
                num_orig_hyps = 1 if steps == 0 else len(hyps_batch[batch])
                all_hyps = []
                for i in xrange(num_orig_hyps):
                    h = hyps_batch[batch][i]
                    #for each of the top 2*beam_size hyps:
                    for j in xrange(int(self.config["BeamSearch"]["beam_size"]) * 2): 
                            # Extend the ith hypothesis with the jth option
                            new_hyp = h.extend(token=topk_ids[batch*beam_size + i, j],
                                        prob=probs[batch*beam_size + i, j],
                                        state_h=states_h_new[batch*beam_size + i],
                                        state_c=states_c_new[batch*beam_size + i],
                                        state_temp_h=states_temp_h_new[batch*beam_size +i],
                                        state_temp_c=states_temp_c_new[batch*beam_size + i],
                                        prev_coverage_vec = prev_coverages_new[batch*beam_size + i ],
                                        prev_coverage_between_decoders = prev_coverage_between_decoders_new[batch*beam_size + i],
                                        prev_coverage_vec_different = prev_coverage_vec_different_new[batch*beam_size + i],
                                        combined_attn_state = combined_attns_states_new[batch*beam_size + i],
                                        attns_state_different=attns_state_different_new[batch*beam_size + i],
                                        attn_values=attns_values[batch*beam_size + i])

                            all_hyps.append(new_hyp)
                #assert(np.all(new_state != states[i]))

                # Filter and collect any hypotheses that have produced the end token.
                hyps = [] # will contain hypotheses for the next step
                for h in sort_hyps(all_hyps): # in order of most likely h
                    if h.latest_token == vocab.word_to_index["<eos>"]: # if stop token is reached...
                        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                        if steps >= int(self.config["Decoder"]["min_dec_steps"]):
                            results[batch].append(h)
                    else: # hasn't reached stop token, so continue to extend this hypothesis
                        hyps.append(h)
                    if len(hyps) == beam_size: 
                        break
             
                hyps_batch[batch]= hyps
        
            steps += 1
            # End of the loop

        # At this point, either we've got beam_size results, or we've reached maximum decoder steps
        for b in range(batch_size):
            if len(results[b])==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
                  results[b] = hyps_batch[b]

        # Sort hypotheses by average log probability
        top_sentences = []
        attention_dists = []

        for batch in xrange(batch_size):
           print ("Batch Number", batch)
           hyps_sorted = sort_hyps(results[batch])
           top_sentences.append(hyps_sorted[0])

        #print (token_predicted)
        decoded_sentences = []
        for i in top_sentences:
           temp = self.dataset.decode_to_sentence(i.tokens[1:])
           print ("Decoder2 ", temp)
           decoded_sentences.append(temp)
           attn_dists = i.attn_dists
           correct_tokens = [np.argmax(k) for k in attn_dists]
           attention_dists.append(correct_tokens)
           

        return decoded_sentences, attention_dists

if __name__ == '__main__':
    run_attention = run_model(sys.argv[1], sys.argv[2],inference=True)
    run_attention.run_inference()
