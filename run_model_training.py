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
    

    def fill_feed_dict(self, encoder_inputs, query_inputs, decoder_inputs,feed_previous=False,feed_true=False):


        """ Fills the feed_dict for training at a given time_step.

            Args:
                encode_inputs : Encoder  sequences
                decoder_inputs : Decoder sequences
                labels : Labels for the decoder
                feed_previous : Whether to pass previous state output to decoder.

            Returns:
                feed_dict : the dictionary created.
        """
        labels = decoder_inputs["word"]


        feed_dict = {
        self.encode_input_word_placeholder : encoder_inputs["word"],
        self.encode_input_maxout_word_placeholder: encoder_inputs["maxout_word"],
        self.encode_input_maxout_vocab_word_placeholder: encoder_inputs["maxout_vocab_word"],
        self.encode_input_char_placeholder : encoder_inputs["char"],
        self.encoder_positional_embedding : encoder_inputs["positional"],
        self.encode_sequence_length: encoder_inputs["seq_length"],
        
        self.query_input_word_placeholder : query_inputs["word"],
        self.query_input_char_placeholder : query_inputs["char"],
        self.query_sequence_length: query_inputs["seq_length"],
        
        
        self.encode_input_content_placeholder: encoder_inputs["content_weights"],
        self.query_position_placeholder: query_inputs["position"],
        self.decode_input_word_placeholder : labels[:,:-1],
        self.label_placeholder : labels[:,1:],
        #self.prop_label_placeholder : decoder_inputs["property_word"][:,:-1],
        self.question_label_placeholder : decoder_inputs["question_label"],
        self.question_position_placeholder : decoder_inputs["question_position"],
        self.weights_placeholder: decoder_inputs["weights"][:,1:],
        self.feed_previous_placeholder: feed_previous,
        self.feed_true_placeholder : feed_true, 
        self.learning_rate_placeholder: self.new_lr,
        self.prev_decoder_label_placeholder : decoder_inputs["property_word"][:,1:],
        self.prev_decoder_input_placeholder : decoder_inputs["property_word"][:,:-1],
        self.prop_indices_placeholder : decoder_inputs["prop_indices"][:,1:],
        self.weights_decoder_label_placeholder : decoder_inputs["property_weights"][:,1:]
        }
        
        if self.config["Hyperparams"]["use_pos_decoder"] == "True":
            feed_dict.update({self.decode_input_pos_placeholder : decoder_inputs["pos"][:,:-1],self.label_pos_placeholder : decoder_inputs["pos"][:,1:]})

        if self.config["GCN"]["use_passage_gcn"] == "True":
            feed_dict.update({self.encode_adj_matrix: encoder_inputs["dep_tree"]})

        if self.config["GCN"]["use_query_gcn"] == "True":
            feed_dict.update({self.query_adj_matrix :  query_inputs["dep_tree"]})

        if not feed_previous:
           feed_dict.update({self.dropout_rate_placeholder: float(self.config["Hyperparams"]["keep_prob"])})

        else:
           feed_dict.update({self.dropout_rate_placeholder: float(1)})
        return feed_dict



    def run_epoch(self, epoch_number, sess, writer,merge,fp = None):

        """ Defines the per epoch run of the model

            Args:
                epoch_number: The current epoch number
                sess:       :  The current tensorflow session.

            Returns
                total_loss : Value of loss per epoch

        """

        start_time = time.time()
        steps_per_epoch = int(math.ceil(float(self.dataset.datasets["train"].number_of_examples) / float(self.config["Hyperparams"]["batch_size"])))

        total_loss = 0

        for step in xrange(steps_per_epoch):

            # Get the next batch


            encoder_inputs, query_inputs, decoder_inputs= self.dataset.next_batch(self.dataset.datasets["train"],int(self.config["Hyperparams"]["batch_size"]), True)


            """ Pass the decoder_inputs for the earlier epochs. As the model
                is trained, the outputs from the previous state should be fed
                to better train the model.
            """
            if (fp is None):
                if(epoch_number > int(self.config["Hyperparams"]["feed_previous"])):
                    feed_previous = True
                else:
                    feed_previous = False

            else:
                feed_previous = fp

            if (epoch_number > 2 and epoch_number % 2 == 0) :
               self.new_lr = self.new_lr
            
            # Feed the placeholders with encoder_inputs,decoder_inputs,decoder_labels
            
            if(epoch_number < int(self.config["Hyperparams"]["feed_true"])):
               feed_true = True
            else:
               feed_true = False

            feed_dict = self.fill_feed_dict(encoder_inputs, query_inputs, decoder_inputs, feed_previous,feed_true=feed_true)
            #merge = tf.summary.merge_all()
            min_kl = 0 
            if epoch_number >= int(self.config["Hyperparams"]["epoch_before_rl"]) and (self.config["Hyperparams"]["rl_plus_crossent"] == "True" or self.config["Hyperparams"]["use_only_rl"] == "True"):
                _,loss_value, summary, loss_reinforce,outputs, _ , bs, bg, coverage_loss, property_coverage_loss, prev_decoder_prop, question_loss, loss_mutual_information,question_gate_loss, question_gates = sess.run([self.train_op, self.loss_op,merge,
                                                                                self.loss_reinforce_op,
                                                                                self.logits, self.attention_weights, self.bleu_sample, self.bleu_greedy, self.coverage_loss, self.property_coverage_loss, self.loss_prev_decoder_prop,self.loss_question_label, self.loss_mutual_information,self.question_gate_loss, self.question_gates], feed_dict=feed_dict)
            else:
                bs = 0
                bg = 0
                _, loss_value,summary,outputs,coverage_loss, property_coverage_loss, prev_decoder_prop,question_loss, loss_mutual_information, masked_weights_p, question_gates  = sess.run([self.train_op_crossent, self.loss_op, merge,
                                                                self.logits,self.coverage_loss,self.property_coverage_loss, self.loss_prev_decoder_prop, self.loss_question_label, self.loss_mutual_information, self.model.model.masked_weights_props, self.question_gates] , feed_dict = feed_dict)
                loss_reinforce = -10.0
            total_loss  += loss_value
            iter_no = epoch_number*steps_per_epoch + step
            writer.add_summary(summary,iter_no)
            #aw = sess.run([self.attention_weights], feed_dict=feed_dict)
            #print ("Attention weights", aw[0])
            #print ("LOGITS", outputs[0], np.argmax(outputs[0]))
            #0print ("Labels", temp_l[:,0])
            duration = time.time() - start_time
            #print (len(outputs))
            #print (outputs[0].shape
            #final_output = np.dstack(outputs)
            #print (final_output[0,...])
            #print (".."*30)
            #print (final_output[1,...])
        
            print ("Epoch {} Step {} Loss {} Reinforce Loss {} BLEU Sample {} BLEU Greedy {} Coverage Loss {} Property Coverage Loss {} Prop Decoder {} Question loss {} MI Loss {} ".format(epoch_number,
                                                                                   step, loss_value,
                                                                                   loss_reinforce, bs, bg,coverage_loss, property_coverage_loss, prev_decoder_prop,question_loss, loss_mutual_information))
            #print ("Step {} Entropy 1 {} Entropy 2 {} ".format(step, entropy_1, entropy_2))
            if (step == 0 and epoch_number ==0 ):
                print('Trainable Variables') 
                print ('\n'.join([v.name for v in tf.trainable_variables()]))

                #for v in tf.trainable_variables():
                #   x_shape = sess.run(v)
                #   print (x_shape.shape)


            #x = sess.run(self.model.grad, feed_dict = feed_dict)
            #print (x)

            sys.stdout.flush()
            # Check the loss with forward propogation
            if (step + 1 == steps_per_epoch ) or ((step  + 1) % int(self.config["Log"]["print_frequency"]) == 0):

                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                sys.stdout.flush()
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.print_titles(sess, self.dataset.datasets["train"], int(self.config["Hyperparams"]["batch_size"]))
                self.saver.save(sess, os.path.join(self.config["Log"]["output_dir"] , 'last_model'))    
                #self.print_titles_in_files(sess, self.dataset.datasets["test"], str(step + 1300*epoch_number))

                # Evaluate against the validation set.
                print('Step %d: loss = %.2f' % (step, loss_value))
                print('Validation Data Eval:')
                #loss_value = self.do_eval(sess,self.dataset.datasets["valid"])
                self.print_titles(sess,self.dataset.datasets["valid"], int(self.config["Hyperparams"]["batch_size"]))
                #print('Step %d: loss = %.2f' % (step, loss_value))
                    
                # Evaluate against the test set.
                #print('Test Data Eval:')
                #loss_value = self.do_eval(sess,self.dataset.datasets["test"])
                #self.print_titles(sess,self.dataset.datasets["test"], 10)
                #print('Step %d: loss = %.2f' % (step, loss_value))

                #self.print_titles_in_files(sess, self.dataset.datasets["test"])
                #self.print_titles_in_files(sess, self.dataset.datasets["train_test"])
                #self.print_titles_in_files(sess, self.dataset.datasets["valid"])
                sys.stdout.flush()

        return float(total_loss)/ float(steps_per_epoch)


    def do_eval(self,sess, data_set):

        """ Does a forward propogation on the data to know how the model's performance is.
             This will be mainly used for valid and test dataset.

            Args:
                sess : The current tensorflow session
                data_set : The datset on which this should be evaluated.

            Returns
                Loss value : loss value for the given dataset.
        """  

        total_loss = 0
        steps_per_epoch =  int(math.ceil(float(data_set.number_of_examples) / float(self.config["Hyperparams"]["batch_size"])))

        for step in xrange(steps_per_epoch): 
            encoder_inputs, query_inputs, decoder_inputs = self.dataset.next_batch(
                data_set,int(self.config["Hyperparams"]["batch_size"]), False)
            
            feed_dict  = self.fill_feed_dict(encoder_inputs, query_inputs, decoder_inputs, feed_previous = True)
            loss_value = sess.run(self.loss_op, feed_dict=feed_dict)
            total_loss += loss_value

        return float(total_loss)/float(steps_per_epoch)



    def print_titles_in_files(self, sess, data_set, epoch = 1000):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """
        total_loss = 0
        f1 = codecs.open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_final_results"+str(epoch)), "w", encoding="utf-8")
        f2 = codecs.open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_attention_weights" +str(epoch)), "w", encoding="utf-8")
        #f3 = codecs.open(os.path.join(self.config["Log"]["output_dir"], data_set.name+ "_output_switches"+str(epoch)), "w", encoding="utf-8")
        
        prev_f1 = codecs.open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_prev_final_results"+str(epoch)), "w", encoding="utf-8")
        prev_f2 = codecs.open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_prev_attention_weights" +str(epoch)), "w", encoding="utf-8")
        #prev_f3 = codecs.open(os.path.join(self.config["Log"]["output_dir"], data_set.name+ "_prev_output_switches"+str(epoch)), "w", encoding="utf-8")
        steps_per_epoch =  int(math.ceil(float(data_set.number_of_examples) / float(self.config["Hyperparams"]["batch_size"])))
        attention_list = [] 
        prev_attention_list = []
        beta_weights_list = []
        alpha_weights_list = []
        attention_list_prop = []
        output_switch_list = [] 
        question_gate_list = []
        attention_weights_different_list = []
        for step in xrange(steps_per_epoch):
            encoder_inputs, query_inputs, decoder_inputs = self.dataset.next_batch(
                data_set,int(self.config["Hyperparams"]["batch_size"]), False)

            feed_dict = self.fill_feed_dict(encoder_inputs, query_inputs, decoder_inputs, feed_previous = True)

            _decoder_states_ , attention_weights,attention_weights_prop,prev_decoder_states,prev_attention_weights,attention_weights_different, output_switch, question_gate = sess.run([self.logits,
            self.attention_weights,self.attention_weights_prop,self.prev_decoder_logits,self.prev_attention_weights,self.attention_weights_different, self.output_switch,self.question_gates], feed_dict=feed_dict)

            if self.config["Encoder"]["use_property_attention"] == "True":
                beta_weights, alpha_weights = sess.run([self.beta_weights, self.property_attention_weights], feed_dict=feed_dict)
                beta_weights_list.append(beta_weights)
                alpha_weights_list.append(alpha_weights) 
            print('attn_wt shape', np.shape(attention_weights))
            print('dec_state shape', np.shape(_decoder_states_)) 
            attention_list_prop.append(attention_weights_prop)
            attention_list.append(attention_weights) 
            prev_attention_list.append(prev_attention_weights)
            attention_weights_different_list.append(attention_weights_different)
            output_switch_list.append(output_switch)
            question_gate_list.append(question_gate)
            #print ('output_switches', np.shape(output_switches))

            #print(attention_weights)
            attention_states = np.array([np.argmax(i,1) for i in attention_weights])
            
            ##  Question Gate #### - [batch_size]
            question_gates = np.asarray(question_gate_list)
            
            # Pack the list of size max_sequence_length to a tensor
            decoder_states = np.array([np.argmax(i,1) for i in _decoder_states_])
            
            #output_switches = np.reshape(output_switches,(64, len(output_switches)))
            # tensor will be converted to [batch_size * sequence_length * symbols]
            train_labels = decoder_inputs["word"]
            ds = np.transpose(decoder_states)
            attention_states = np.transpose(attention_states)
            assert(len(ds) == len(attention_states))
            attn_state = attention_states 
            true_labels = train_labels
            # Converts this to a length of batch sizes
            final_ds = ds.tolist()
            final_as = attn_state.tolist()
            true_labels = true_labels.tolist()
            #output_switches = output_switches.tolist()
            #print(final_ds)
            
           #####################  PREV Decoder  ##########################        
  
            prev_attention_states = np.array([np.argmax(i,1) for i in prev_attention_weights])
            # Pack the list of size max_sequence_length to a tensor
            prev_decoder_states = np.array([np.argmax(i,1) for i in prev_decoder_states])
            #output_switches = np.reshape(output_switches,(64, len(output_switches)))
            # tensor will be converted to [batch_size * sequence_length * symbols]
            prev_train_labels = decoder_inputs["property_word"]
            prev_ds = np.transpose(prev_decoder_states)
            prev_attention_states = np.transpose(prev_attention_states)
            assert(len(prev_ds) == len(prev_attention_states))
            prev_attn_state = prev_attention_states 
            prev_true_labels = prev_train_labels
            # Converts this to a length of batch sizes
            prev_final_ds = prev_ds.tolist()
            prev_final_as = prev_attn_state.tolist()
            prev_true_labels = prev_true_labels.tolist()
            #output_switches = output_switches.tolist()
            #print(final_ds)
            
           ######################################################################
            
            for i, (states,prev_states) in enumerate(zip(final_ds,prev_final_ds)):
                # Get the index of the highest scoring symbol for each time step
                #indexes = sess.run(tf.argmax(states, 1))
                s =  self.dataset.decode_to_sentence(states)
                t =  self.dataset.decode_to_sentence(true_labels[i])
                f1.write(s + "\n")
                f1.write(t +"\n")
                x = " ".join(str(m) for m in final_as[i])
                #y = " ".join(str(m) for m in output_switches[i])
                f2.write(x + "\n")
                #f3.write(y + "\n")
                
                #########################  Prev Decoder ###############################

                # Get the index of the highest scoring symbol for each time step
                #indexes = sess.run(tf.argmax(states, 1))
                prev_s =  self.dataset.decode_to_sentence(prev_states)
                prev_t =  self.dataset.decode_to_sentence(prev_true_labels[i])
                prev_f1.write(prev_s + "\n")
                prev_f1.write(prev_t +"\n")
                prev_x = " ".join(str(m) for m in prev_final_as[i])
                #y = " ".join(str(m) for m in output_switches[i])
                prev_f2.write(prev_x + "\n")
                #f3.write(y + "\n") 

                #########################################################################
        
        # pickle.dump(attention_list,open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_attention_pickle_"+str(epoch)),"wb"))
        # pickle.dump(attention_list_prop,open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_attention_prop_pickle_"+str(epoch)),"wb"))
        # pickle.dump(prev_attention_list,open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_prev_attention_pickle_"+str(epoch)),"wb")) 
        # pickle.dump(output_switch_list,open(os.path.join(self.config["Log"]["output_dir"],data_set.name+"_output_switch_pickle_"+str(epoch)),"wb")) 
        
        if data_set.name == "test":

            ###################  Running Bleu Score Evaluation Script for the 2 decoders ##########################
            script_name = "extract_labels.sh"
            script_name = os.path.join(os.getcwd(),"postprocess_scripts",script_name)
            data_dir = sys.argv[1]
            std_out = subprocess.Popen([script_name, self.config["Log"]["output_dir"],data_dir,"True","False",str(epoch)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
            temp_output = [ i for i in std_out.stdout.readlines() ]
            temp_err = [ i for i in std_out.stdout.readlines() ]
            #test_bleu = compute_metrics(os.path.join(self.config["Log"]["output_dir"], "test_final_results" + str(epoch) + "_plabels_copy"), [os.path.join(data_dir, "test_summary_lower")])
            print (temp_output,temp_err)
            test_bleu = float(temp_output[-3].split(" ")[1])
    
            print ("Output Error",temp_output,temp_err)
            script_name = "extract_labels_prev_same.sh"
            script_name = os.path.join(os.getcwd(),"postprocess_scripts",script_name)
            data_dir = sys.argv[1]
            std_out = subprocess.Popen([script_name, self.config["Log"]["output_dir"],data_dir,"True","False",str(epoch)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
            #test_prev_bleu = compute_metrics(os.path.join(self.config["Log"]["output_dir"], "test_prev_final_results" + str(epoch) + "_plabels_copy"), [os.path.join(data_dir, "test_summary_lower")])
            temp_output = [ i for i in std_out.stdout.readlines() ]
            temp_err = [ i for i in std_out.stdout.readlines() ]
            print ("Output Error",temp_output,temp_err)
            test_prev_bleu = float(temp_output[-3].split(" ")[1])
    
            print ("{} Epoch: {} 2nd Decoder score: {} 1st Decoder score: {}".format(data_set.name, epoch, test_bleu, test_prev_bleu))
            ##########################  Attention Plotting  ##############################################
        
            max_count = 50
    
            y2 = os.path.join(self.config["Log"]["output_dir"],data_set.name+"_final_results"+str(epoch)+"_plabels_copy")
            x = os.path.join(data_dir,data_set.name + "_content")
            out = os.path.join(self.config["Log"]["output_dir"],data_set.name+"_attn_plots_ed2")
            save_plot(attention_list,y2,x,max_count, out, epoch) 
            
            y3 = os.path.join(self.config["Log"]["output_dir"],data_set.name+"_final_results"+str(epoch)+"_plabels_copy")
            x = os.path.join(data_dir,data_set.name + "_content")
            out = os.path.join(self.config["Log"]["output_dir"],data_set.name+"_attn_plots_ed2diff")
            save_plot(attention_weights_different_list,y3,x,max_count, out, epoch) 

            y1 = os.path.join(self.config["Log"]["output_dir"],data_set.name+"_prev_final_results"+str(epoch)+"_plabels_copy")
            out = os.path.join(self.config["Log"]["output_dir"],data_set.name+"_attn_plots_ed1")
            save_plot(prev_attention_list,y1,x,max_count, out, epoch)

            out = os.path.join(self.config["Log"]["output_dir"],data_set.name+"_attn_plots_d1d2")
            save_plot(attention_list_prop,y2,y1,max_count, out, epoch)      

            return test_bleu, test_prev_bleu
        else:
            return 0, 0

    
    def print_titles(self, sess, data_set, total_examples):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """

        encoder_inputs, query_inputs, decoder_inputs = self.dataset.next_batch(
            data_set, total_examples, False)

        feed_dict = self.fill_feed_dict(encoder_inputs, query_inputs, decoder_inputs, feed_previous = True)

        _decoder_states_, = sess.run([self.logits], feed_dict=feed_dict)

        # Pack the list of size max_sequence_length to a tensor
        decoder_states = np.array([np.argmax(i,1) for i in _decoder_states_])

        #predicted_pos_labels = np.transpose(np.array([np.argmax(i,1) for i in pos_projection])).tolist()
        #true_pos_labels = decoder_inputs["pos"].tolist()

        train_labels = decoder_inputs["word"]
        query_labels = query_inputs["word"]
        passage_labels = encoder_inputs["word"]
        # tensor will be converted to [batch_size * sequence_length * symbols]
        ds = np.transpose(decoder_states)
        true_labels = train_labels

        # Converts this to a length of batch size
        final_ds = ds.tolist()
        true_labels = true_labels.tolist()
        query_labels = query_labels.tolist()
        passage_labels = passage_labels.tolist()
        for i,states in enumerate(final_ds):

            # Get the index of the highest scoring symbol for each time step
            #indexes = sess.run(tf.argmax(states, 1))
            print (true_labels[i])
            print ('<=>'*20)
            print (states)
            print ('<=>'*20)

            print ("Title is " , self.dataset.decode_to_sentence(states).encode('utf-8'))
            print ("True Summary is " , self.dataset.decode_to_sentence(true_labels[i]).encode('utf-8'))
            print ("Query is " , self.dataset.decode_to_sentence(query_labels[i]).encode('utf-8'))
            print ("Passage is " , self.dataset.decode_to_sentence(passage_labels[i]).encode('utf-8'))
            #print ("Predicted POS Labels ",self.dataset.decode_to_pos_tags(predicted_pos_labels[i]).encode('utf-8'))
            #print ("True POS Labels ",self.dataset.decode_to_pos_tags(true_pos_labels[i]).encode('utf-8'))


    def train_op_function(self, loss, learning_rate, gradient_clip):
        if self.config["Hyperparams"]["optimizer"] == "SGD":
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate)

        regularizing_loss = [[tf.nn.l2_loss(var)] for var in tf.trainable_variables()]
        regularizing_loss = tf.reduce_mean(tf.concat(regularizing_loss, axis=-1))
        regularizing_loss = float(self.config["Hyperparams"]["l2_lambda"]) * regularizing_loss
        train_op = optimizer.minimize(loss + regularizing_loss)
        grads = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value( grad, -1 * gradient_clip, 1* gradient_clip), var) for grad, var in grads if grad is not None]
        train_op = optimizer.apply_gradients(capped_gvs)

        return train_op

    def build_graph(self):


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
                                   "char": self.encode_input_char_placeholder,
                                   "seq_length": self.encode_sequence_length,
                                   "positional": self.encoder_positional_embedding,
                                   "dep_tree":self.encode_adj_matrix, 
                                   "content_weights": self.encode_input_content_placeholder, 
                                   "dropout": self.dropout_rate_placeholder,
                                   "maxout_word": self.encode_input_maxout_word_placeholder,
                                    "maxout_vocab_word": self.encode_input_maxout_vocab_word_placeholder

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



            final_outputs  = self.model.pipeline(self.encoder_input_batch, self.query_input_batch, self.decoder_input_batch,
                                                   self.feed_previous_placeholder,self.feed_true_placeholder)

            self.logits, self.attention_weights, self.logits_pos = final_outputs.get("outputs"), final_outputs.get("attention_weights"), final_outputs.get("pos_outputs")
            self.beta_weights, self.property_attention_weights = final_outputs["beta_values"], final_outputs["property_attention_weights"]
            self.logits_rein = final_outputs.get("outputs")
            self.bleu_greedy = final_outputs.get("bleu_crossent")
            self.bleu_sample = final_outputs.get("bleu_reinforce")
            self.question_gate_loss = final_outputs.get("question_gate_loss")
            self.question_gate_rewards = final_outputs.get("question_gate_rewards")
            self.predictions_reinforce = final_outputs.get("predictions")
            self.question_logits = final_outputs["question_logits"]
            # self.initial_values = final_outputs["initial_values"]
            self.new_state_values = final_outputs["new_state_values"]

            self.prev_decoder_logits = final_outputs["prev_decoder_outputs"]
            self.prev_decoder_tokens = final_outputs["prev_decoder_tokens"]            
            self.prev_attention_weights = final_outputs["prev_attention_weights"] 
            self.prev_output_switch = final_outputs["prev_output_switch"] 
            self.question_gates = final_outputs["question_gates"]
            self.attention_weights_prop = final_outputs["attention_weights_prop"]
            self.attention_weights_different = final_outputs["new_state_values"]["attention_values_different"]

            print ("LOGITS", self.new_state_values)
           
            # Add to the Graph the Ops for loss calculation.
            self.new_lr = float(self.config["Hyperparams"]["learning_rate"])
            self.loss_op = loss_op(self.logits, self.label_placeholder, self.weights_placeholder, is_copy = (self.config["Decoder"]["decoder_type"] == "copy"), 
                          batch_size = int(self.config["Hyperparams"]["batch_size"]))


            total_loss = 0 
            temp_lambda =  float(self.config["Aux_Loss"]["last_decoder_loss_lambda"])
            total_loss = temp_lambda*self.loss_op 

            self.coverage_loss = tf.zeros(1)
            self.property_coverage_loss = tf.zeros(1)
            self.loss_qbleu_label = tf.zeros(1)
            self.loss_prev_decoder_prop = tf.zeros(1)
            #self.config["Hyperparams"]["coverage_lambda"]) = str(0)
            #self.config["Hyperparams"]["coverage_property_lambda"]) = str(0)

            if self.config["Aux_Loss"]["prev_decoder_prop"] == "True":
                self.loss_prev_decoder_prop = loss_op(self.prev_decoder_logits, self.prev_decoder_label_placeholder, self.weights_decoder_label_placeholder, is_copy= (self.config["Decoder"]["decoder_type"] == "copy"), batch_size=int(self.config["Hyperparams"]["batch_size"]))
                total_loss += float(self.config["Aux_Loss"]["prev_decoder_prop_lambda"]) * self.loss_prev_decoder_prop
            
            self.loss_mutual_information = tf.zeros([1])

            self.entropy_1 = entropy(self.prev_decoder_logits,  self.weights_placeholder)
            self.entropy_2 = entropy(self.logits,  self.weights_placeholder)
            if self.config["Aux_Loss"]["mutual_information"] == "True":
                self.loss_mutual_information = information_gain_between_decoders(self.logits, self.weights_placeholder, 
                                                                                 self.prev_decoder_logits, self.weights_placeholder)
                total_loss += float(self.config["Aux_Loss"]["mutual_information_lambda"]) * self.loss_mutual_information


            if self.config["Aux_Loss"]["use_prop_label"] == "True":
                print("beta_weights_shape", len(self.beta_weights))
                beta_weights_temp = [i[0] for i in self.beta_weights]
                dummy_row = tf.ones(shape=(int(self.config["Hyperparams"]["batch_size"]), 1))*(1)
                beta_weights_temp = [tf.concat([i, dummy_row], axis=-1) for i in beta_weights_temp]
                #beta_weights_temp = tf.concat([beta_weights_temp, dummy_row], axis=-1)
                self.loss_qbleu_label = loss_op(beta_weights_temp, self.prop_label_placeholder, self.weights_placeholder, is_copy= (self.config["Decoder"]["decoder_type"] == "copy"),
                                              batch_size = int(self.config["Hyperparams"]["batch_size"]))
                total_loss += float(self.config["Aux_Loss"]["prop_label_lambda"])*self.loss_qbleu_label

            if self.config["Decoder"]["use_coverage"] == "True":
                self.coverage_loss =  loss_coverage(self.attention_weights,int(self.config["Hyperparams"]["batch_size"]))
                self.coverage_loss += loss_coverage(self.prev_attention_weights, int(self.config["Hyperparams"]["batch_size"]))
                total_loss += float(self.config["Hyperparams"]["coverage_lambda"])*self.coverage_loss           

            self.min_kl = tf.zeros(1) 
            if self.config["Encoder"]["use_property_attention"] == "True":
                if self.config["Encoder"]["use_js"] == "True":
                    self.property_coverage_loss = -1*loss_js(self.property_attention_weights,int(self.config["Hyperparams"]["batch_size"]))
                    self.min_kl = loss_prop_aw(self.property_attention_weights, self.attention_weights, int(self.config["Hyperparams"]["batch_size"]))
                    total_loss += float(self.config["Hyperparams"]["js_lambda"])* (self.property_coverage_loss + self.min_kl)
                else:
                    self.property_coverage_loss =  loss_coverage(self.property_attention_weights,int(self.config["Hyperparams"]["batch_size"]))
                    total_loss += float(self.config["Hyperparams"]["coverage_property_lambda"])* self.property_coverage_loss 
            
            self.loss_question_label = tf.zeros(1)
            if self.config["Aux_Loss"]["double_stocastic_attention"] == "True":
                temp_attention_weights = tf.stack(self.attention_weights_prop, axis=-1)
                temp_attention_weights = 1- tf.reduce_sum(temp_attention_weights, axis=-1)
                temp_attention_weights = temp_attention_weights * temp_attention_weights*self.weights_decoder_label_placeholder
                temp_attention_weights = tf.reduce_sum(temp_attention_weights, axis=-1)
                temp_attention_weights = tf.reduce_mean(temp_attention_weights)
                self.doubly_stocastic_loss = float(self.config["Aux_Loss"]["double_stocastic_attention_lambda"]) * temp_attention_weights
                total_loss += self.doubly_stocastic_loss

            self.output_switch = final_outputs["output_switch"]

            if not self.config["Decoder"].get("no_prop_copy") == "True":   

                self.output_switch_final = tf.unstack(tf.expand_dims(tf.stack(final_outputs["output_switch"],axis=0),axis=3),axis=2)
                self.attention_weights = self.output_switch_final[0] * self.attention_weights
                self.attention_weights_prop = self.output_switch_final[1]  * self.attention_weights_prop

	    self.prop_indices_memnet_loss = tf.zeros(1)
            if self.config["Aux_Loss"]["prop_indices_loss"] == "True":
                temp_output_switch_final  = [tf.expand_dims(i[:,1], axis=1) for i in self.output_switch]
                temp_attention_weights_prop = tf.unstack(self.attention_weights_prop, axis=0)
                temp_attention_weights_prop = [tf.concat([temp_attention_weights_prop[m], 1 - temp_output_switch_final[m]], axis=-1) for m in  range(19)]
                print (temp_attention_weights_prop)
                self.prop_indices_memnet_loss = loss_op(temp_attention_weights_prop, self.prop_indices_placeholder, self.weights_placeholder, is_copy=(self.config["Decoder"]["decoder_type"] == "copy"), batch_size = int(self.config["Hyperparams"]["batch_size"]))
                total_loss += float(self.config["Aux_Loss"]["prop_indices_lambda"]) * self.prop_indices_memnet_loss

            if self.config["Hyperparams"]["use_only_rl"] == "True" or self.config["Hyperparams"]["rl_plus_crossent"] == "True":
                self.loss_reinforce_op = reinforce_loss(self.config,self.logits_rein, self.predictions_reinforce,self.bleu_sample, self.bleu_greedy, self.weights_placeholder, 
                                                        batch_size = int(self.config["Hyperparams"]["batch_size"]))
                self.loss_reinforce_op_reverse = reinforce_loss(self.config,self.prev_decoder_logits, self.prev_decoder_tokens, self.bleu_greedy, self.bleu_sample, self.weights_placeholder, 
                                                        batch_size = int(self.config["Hyperparams"]["batch_size"]))

            if self.config["Hyperparams"]["use_only_rl"] == "True":
                total_loss_rl = self.loss_reinforce_op
            elif self.config["Hyperparams"]["rl_plus_crossent"] == "True":
                total_loss_rl  = (1 - self.rl_lambda_value) * total_loss + (self.rl_lambda_value) * self.loss_reinforce_op            
                if self.config["Hyperparams"]["use_reverse_rl"] == "True":
                    total_loss_rl  = (1 - self.rl_lambda_value) * total_loss + (self.rl_lambda_value)/2.0 * (self.loss_reinforce_op + self.loss_reinforce_op_reverse)
                elif self.config["Hyperparams"]["use_only_reverse_rl"] == "True":
                    total_loss_rl  = (1 - self.rl_lambda_value) * total_loss + (self.rl_lambda_value)* (self.loss_reinforce_op)

            if self.config["Hyperparams"]["use_only_rl"] == "True" or self.config["Hyperparams"]["rl_plus_crossent"] == "True" :    
                # Add to the Graph the Ops that calculate and apply gradients.
                self.train_op = self.train_op_function(total_loss_rl, self.learning_rate_placeholder, float(self.config["Hyperparams"]["gradient_clip"]))
                 

            self.train_op_crossent = self.train_op_function(total_loss, self.learning_rate_placeholder, float(self.config["Hyperparams"]["gradient_clip"]))
            # Add the variable initializer Op.
            total_parameters = count_parameters()
            sys.stdout.flush()

            print ("<=>"*30)
            print ("Total Number of Parameters",total_parameters)


    def run_training(self):
            
    
        with tf.Graph().as_default():
    
            self.build_graph()
            init = tf.initialize_all_variables()
            print ("Init done")
         
            # Create a saver for writing training checkpoints.
            #c = tf.ConfigProto()
            #c.gpu_options.allow_growth = True
            #session = tf.Session()

            # Create a session for running Ops on the Graph.
            
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            sess = tf.Session(config=config)

            tb_path = os.path.join(self.config["Log"]["output_dir"] ,  "tensorboard")
            if not (os.path.exists(tb_path)):
                os.makedirs(tb_path)
            writer = tf.summary.FileWriter(tb_path ,sess.graph)

            # Instantiate a SummaryWriter to output summaries and the Graph.

            sess.run(init)
            saver = tf.train.Saver()
            self.saver = saver
            # if best_model exists pick the weights from there:
            if (os.path.exists(os.path.join(self.config["Log"]["output_dir"] , "best_model.meta"))):
                print ("Initializing the model with the best saved model")
                optimistic_restore(sess, os.path.join(self.config["Log"]["output_dir"] , "best_model"))
                best_test_bleu, best_test_bleu_prev = self.print_titles_in_files(sess, self.dataset.datasets["valid"])

                saver.save(sess, os.path.join(self.config["Log"]["output_dir"], "saved_params"))
                #self.print_titles_in_files_beamsearch(sess, self.dataset.datasets["test"], self.dataset.vocab, 101)
                if (sys.argv[3] == "plots"):
		        	return 
                 
            else:
                best_val_loss = float('inf')
                best_test_bleu = 0.0
                best_test_bleu, best_test_bleu_prev = self.print_titles_in_files(sess, self.dataset.datasets["valid"])
            
            if (os.path.exists(os.path.join(self.config["Log"]["output_dir"],  "best_model.meta"))):
                print ("Initializing the model with the last saved epoch")
                optimistic_restore(sess, os.path.join(self.config["Log"]["output_dir"] , "best_model"))

            else:
                # Run the Op to initialize the variables.
                sess.run(init)
                best_test_bleu = 0
                best_test_bleu_prev = 0
                #best_test_bleu,best_test_bleu_prev = self.print_titles_in_files(sess, self.dataset.datasets["test"], 1111)

            # To store the model that gives the best result on validation.
            best_test_epoch = 0

            merge = tf.summary.merge_all()
            for epoch in xrange(int(self.config["Hyperparams"]["max_epochs"])):

                print ("Epoch: " + str(epoch))
                start = time.time()
                train_loss = self.run_epoch(epoch, sess,writer,merge)
                valid_loss = self.do_eval(sess, self.dataset.datasets["valid"])

                print ("Loss {}".format(train_loss))
                 
                #if valid_loss <= best_val_loss:
                #    best_val_loss = valid_loss
                #    best_val_epoch = epoch
                #    saver.save(sess, os.path.join(self.config["Log"]["output_dir"] , 'best_model'))

                #if (epoch == self.config.max_epochs - 1):
                saver.save(sess, os.path.join(self.config["Log"]["output_dir"] , 'last_model'))

                valid_bleu, valid_bleu_prev = self.print_titles_in_files(sess, self.dataset.datasets["valid"], epoch)
                                
                if valid_bleu >= best_test_bleu:
                    best_test_bleu = valid_bleu
                    best_test_epoch = epoch
                    print ("Saving best_model at epoch",epoch)
                    saver.save(sess, os.path.join(self.config["Log"]["output_dir"] , 'best_model'))

                if valid_bleu_prev >= best_test_bleu_prev:
                    best_test_bleu_prev = valid_bleu_prev
                    print ("Saving best_model_prev at epoch",epoch)
                    saver.save(sess, os.path.join(self.config["Log"]["output_dir"] , 'best_model_prev'))
                
                if (epoch - best_test_epoch > int(self.config["Hyperparams"]["early_stop"])):
                    print ("Results are getting no better. Early Stopping")
                    break

                print ("Epoch: {} Training Loss: {} Validation Loss: {} Total time:{}".format(epoch, train_loss, valid_loss, time.time() - start))

            optimistic_restore(sess, os.path.join(self.config["Log"]["output_dir"] , 'best_model'))
            test_loss = self.do_eval(sess, self.dataset.datasets["test"])

            #self.print_titles_in_files(sess, self.dataset.datasets["test"])
            #self.print_titles_in_files(sess, self.dataset.datasets["valid"])
            self.print_titles_in_files_beamsearch(sess,  self.dataset.datasets["test"], self.dataset.vocab)
            self.print_titles_in_files_beamsearch(sess,  self.dataset.datasets["valid"], self.dataset.vocab)

	    #print ("Test Loss:{}".format(test_loss))
def main():
    run_attention = run_model(sys.argv[1], sys.argv[2])
    run_attention.run_training()

if __name__ == '__main__':
    main()
