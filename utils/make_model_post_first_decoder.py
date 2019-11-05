import tensorflow as tf
from tensorflow.python.util import nest

import numpy as np
import rnn_cell
import collections
import math
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import answerability_score
import random
from model import *
from loss import * 

class make_model:
    def __init__(self, args, embedding_initializer = None, vocab_length = None, vocab_char_length = None, vocab_dict=None, dropout_param=None):
        self.vocab_length = vocab_length
        self.vocab_dict = vocab_dict
        self.dropout_param= dropout_param

        if args["Hyperparams"]["use_qbleu"] == "True":
            answerability_score.word_to_index = vocab_dict.index_to_word

        if "use_dynamic_score" in args["Hyperparams"] and args["Hyperparams"]["use_dynamic_score"] == "True":
            dynamic_score.index_to_word = vocab_dict.index_to_word
        
        self.model = BasicSeq2SeqWithAttentionwithQuery(args, embedding_initializer= embedding_initializer, vocab_length=vocab_length, vocab_char_length=vocab_char_length,
                                                        vocab_dict=vocab_dict, dropout_param=dropout_param)

    def encoder_module(self, encoder_input_batch, query_input_batch):
	
	    with tf.variable_scope("encoder_module",reuse=tf.AUTO_REUSE):
	        batch_size = int(self.model.args["Hyperparams"]["batch_size"]) 
	        prop = None
	        question_logits = []
	        
	        
	        encoder_outputs, encoder_state = self.model.passage_encoder(encoder_input_batch)
	
	       	print ("ENCODER OUTPUTS SHAPE before self attention",encoder_outputs.get_shape()) 
	        
	        
	        if bool(self.model.args["Query"]["use_query"] == "True" and self.model.args["Query"]["use_position"] == "True"):
	
	            temp_batches = tf.reshape(tf.range(0, limit=batch_size), shape=(-1,1))
	            temp_batches = tf.tile(temp_batches, multiples=[1, int(self.model.args["Query"]["max_sequence_length"])])
	            print ("temp_batches shape",temp_batches.get_shape())
	            print ("query_input_batch position shape",query_input_batch["position"].get_shape())
	
	            temp_query_indices = tf.stack([temp_batches, query_input_batch["position"]],axis=2)
	            temp_query_embeddings = tf.gather_nd(encoder_outputs, temp_query_indices)
	            out_size = temp_query_embeddings.get_shape()[-1].value #+ query_outputs.get_shape()[-1].value
	            seq_length = 10#query_outputs.get_shape()[1].value
	
	            query_outputs, query_state = self.model.query_encoder(query_input_batch, temp_query_embeddings)
	
	        elif self.model.args["Query"]["use_query"] == "True":
	            query_outputs, query_state = self.model.query_encoder(query_input_batch)
	
	        else:
	            query_outputs, query_state = tf.zeros([1]),tf.zeros([1])
	
	        #orig_encoder_outputs = encoder_outputs
	        if self.model.args["Query"]["use_query"] == "True" and self.model.args["Encoder"]["use_coattention"] == "True":
	            with tf.variable_scope("coattn_query", reuse=tf.AUTO_REUSE):
	                 query_state_temp = tf.nn.dropout(tf.contrib.layers.fully_connected(query_state, encoder_outputs.get_shape()[-1].value, activation_fn=None), keep_prob=self.model.dropout_param)
	                 query_state_temp = tf.tile(tf.expand_dims(query_state_temp, axis=1), multiples=[1, encoder_outputs.get_shape()[1].value, 1])
	                 new_x = tf.concat([encoder_outputs, query_state_temp, encoder_outputs * query_state_temp], axis=-1)
	            with tf.variable_scope("coattention", reuse=tf.AUTO_REUSE):
	                 encoder_outputs =  tf.nn.dropout(tf.contrib.layers.fully_connected(new_x, encoder_outputs.get_shape()[-1].value, activation_fn=tf.nn.tanh), keep_prob=self.model.dropout_param)
	            
	
	        if "use_gated_self_attention" in self.model.args["Encoder"] and self.model.args["Encoder"]["use_gated_self_attention"] == "True":
	            encoder_outputs = self.model.gated_self_attention("self_attention",encoder_outputs, masked_weights = encoder_input_batch["word"])
	            encoder_state = tf.reduce_mean(encoder_outputs, axis=1)
	        
	
	        #if "use_gated_three_level_self_attention" in self.model.args["Encoder"] and self.model.args["Encoder"]["use_gated_three_level_self_attention"] == "True":
	         #   print  ("Original Encoder Outputs", orig_encoder_outputs, encoder_outputs)
	          #  encoder_outputs = self.model.gated_three_level_self_attention("gated_three_self_attention",orig_encoder_outputs, encoder_outputs, masked_weights = encoder_input_batch["word"])
	
	        if self.model.args["Encoder"]["use_property_attention"] == "True":
	            num_heads = int(self.model.args["Encoder"]["num_property_values"])
	        else:
	            num_heads = int(self.model.args["Attention"]["num_heads"])
	
	        if self.model.args["Query"]["use_query"] == "True" and self.model.args["Encoder"]["use_property_attention"] == "True":
	
	           temp_prop_query = tf.split(query_state, int(self.model.args["Encoder"]["num_property_values"]), axis=-1)
	           new_memory_states, new_attention_weights = self.model.compute_multi_head_attention("property", encoder_outputs, encoder_input_batch["word"], query_state, return_all=True,
	                                                      num_heads=num_heads, property_coverage=True, property_query_states = temp_prop_query)
	
	           self.model.property_attention_weights = new_attention_weights
	           prop = new_attention_weights
	
	           # Stacked just to unpack things in decoder function
	           new_memory_states_temp = tf.stack(new_memory_states, axis=1)
	           encoder_outputs = tf.concat([encoder_outputs, new_memory_states_temp], axis=1)
	           # Classify into 8 of the question type
	
	           with tf.variable_scope("QuestionLabelClassification", reuse=tf.AUTO_REUSE):
	               question_logits = tf.contrib.layers.fully_connected(new_memory_states[0], 9, activation_fn=None)               
	
	        elif self.model.args["Query"]["use_query"] == "True":
	            question_logits = tf.contrib.layers.fully_connected(query_state, 9, activation_fn=None)
	        else:
	            question_logits = tf.zeros([1])
	
	        return encoder_outputs, encoder_state, query_outputs, query_state, prop, question_logits


    def pre_pipeline(self,  encoder_input_batch, query_input_batch, decoder_input_batch, feed_previous, feed_true):
        self.model.dropout_param = encoder_input_batch["dropout"]

        self.encoder_outputs, self.encoder_state, self.query_outputs, self.query_state, self.prop, self.question_logits  = self.encoder_module(encoder_input_batch, query_input_batch)
        print ("encoder shape ---- ",self.encoder_outputs.get_shape())
        self.model.memory_states = self.encoder_outputs
        self.model.query_state = self.query_state
        self.model.encoder_state = self.encoder_state
        self.model.query_outputs = self.query_outputs
        self.model.masked_weights = encoder_input_batch["seq_length"]
        self.model.masked_weights_props = decoder_input_batch["property_labels"]
        self.initial_states =  self.model._init_decoder_params(distract_shape=self.model.memory_states.get_shape()[-1].value, is_prev_state=None, var_name="")


    #         initial_states = self.model._init_decoder_params(self.model.memory_states.get_shape()[-1].value, is_prev_state=is_prev_state, var_name=var_name)
    def post_first_decoder(self, prev_decoder_tokens, prev_attention_weights, query_state):

        
        # get the sequence length of the generated question. remove everthing after first <eos>
        question_gate = tf.zeros(shape=[int(self.model.args["Hyperparams"]["batch_size"])])
        question_gate_policy = tf.zeros(shape=[int(self.model.args["Hyperparams"]["batch_size"])])

        #prev_decoder_tokens = tf.zeros(shape=[int(self.model.args["Hyperparams"]["batch_size"]), 19], dtype=tf.int32)
        final_embedding  = tf.nn.embedding_lookup(self.model.word_embeddings, prev_decoder_tokens)


        print (final_embedding.get_shape())
        prev_decoder_seq_length =  tf.argmax(tf.cast(tf.equal(prev_decoder_tokens, 3*tf.ones_like(prev_decoder_tokens)), tf.int32), axis=-1)
        prev_decoder_seq_length = tf.where(tf.equal(prev_decoder_seq_length, tf.zeros_like(prev_decoder_seq_length)),
                                    (int(self.model.args["Decoder"]["max_sequence_length"]) - 1)*tf.ones_like(prev_decoder_seq_length), prev_decoder_seq_length) + 1   # one is added to unmask eos.
        prev_decoder_seq_length = tf.stop_gradient(prev_decoder_seq_length)

        # for d1-d2 attention mechanism , we need not pay attention on the tokens after first <eos>. masked_weights_props stores these weights.
        prev_decoder_seq_length_tile = tf.tile(tf.expand_dims(prev_decoder_seq_length,axis=1),[1,prev_decoder_tokens.get_shape()[1].value])
        indices =  tf.tile(tf.expand_dims(tf.range(prev_decoder_tokens.get_shape()[1].value,dtype=tf.int64),axis=0),[int(self.model.args["Hyperparams"]["batch_size"]),1])
        prev_decoder_masked_weights = tf.where(tf.less(indices,prev_decoder_seq_length_tile),tf.ones_like(indices),tf.zeros_like(indices))
        masked_weights_props = tf.stop_gradient(prev_decoder_masked_weights)

        # don't add attention distributions(encoder-d1) for tokens after first <eos>
        question_mask = tf.cast(masked_weights_props,tf.float32)
        total_encoder_teacher_attention_weights = tf.reduce_sum(tf.stack(prev_attention_weights, axis=1)*tf.expand_dims(question_mask,axis=2), axis=1)

        #total_encoder_teacher_attention_weights  = tf.zeros(shape=[int(self.model.args["Hyperparams"]["batch_size"]), 100])
        print ("IIIIII", prev_attention_weights)
        prev_decoder_mean_attn = total_encoder_teacher_attention_weights/ tf.reduce_sum(question_mask,axis=1,keepdims=True)



        final_embedding = tf.stop_gradient(final_embedding)
        prev_decoder_mean_attn = tf.stop_gradient(prev_decoder_mean_attn)
        total_encoder_teacher_attention_weights = tf.stop_gradient(total_encoder_teacher_attention_weights)
        
        # BiLSTM to encode the question words
        if self.model.args["Question_encoder"]["use_question_encoder"] == "True":
            with tf.variable_scope("question_encoder",reuse=tf.AUTO_REUSE):
                question_outputs, question_state = self.model.question_encoder(final_embedding,prev_decoder_seq_length)
                question_gate = self.model.gate_soft(question_state)
                #question_outputs = tf.contrib.layers.fully_connected(question_outputs,final_embedding[0].get_shape()[-1].value)
                if self.model.args["Question_encoder"]["use_gate_policy"] == "True":
                    if self.model.args["Question_encoder"]["use_rl_on_gate"] == "True":
                        question_gate, question_gate_policy = self.model.gate_policy(question_state)
                        question_gate = tf.cast(question_gate,tf.float32) 
                    
                    else:
                        question_gate = self.model.gate_soft(question_state)

            #  Attending on the question based on the answer
            with tf.variable_scope("coattn_question", reuse=tf.AUTO_REUSE):
                query_state_temp = tf.nn.dropout(tf.contrib.layers.fully_connected(query_state, question_outputs.get_shape()[-1].value, activation_fn=None), keep_prob=self.model.dropout_param)
                query_state_temp = tf.tile(tf.expand_dims(query_state_temp, axis=1), multiples=[1, question_outputs.get_shape()[1].value, 1])
                new_x = tf.concat([question_outputs, query_state_temp, question_outputs * query_state_temp], axis=-1)
            with tf.variable_scope("coattention_question", reuse=tf.AUTO_REUSE):
                question_outputs =  tf.nn.dropout(tf.contrib.layers.fully_connected(new_x, question_outputs.get_shape()[-1].value, activation_fn=tf.nn.tanh), keep_prob=self.model.dropout_param)

                # if word concat is true then pass both contextual and word embedding
                if self.model.args["Question_encoder"]["word_concat"] == "True":
                    final_embedding = tf.concat([question_outputs, final_embedding], axis=-1)
                else:
                    final_embedding = question_outputs

        return final_embedding,  masked_weights_props, prev_decoder_mean_attn, total_encoder_teacher_attention_weights, question_gate, question_gate_policy


    def pipeline(self,  encoder_input_batch, query_input_batch, decoder_input_batch, feed_previous, feed_true):
            
        # Dropout placeholder
        self.model.dropout_param = encoder_input_batch["dropout"]
        self.model.is_train = tf.cast(1- tf.cast(feed_previous, tf.int32), tf.bool)
        self.pre_pipeline(encoder_input_batch, query_input_batch, decoder_input_batch, feed_previous, feed_true)

        max_steps = int(self.model.args["Decoder"]["max_decoder_steps"]) - 1
        final_outputs = {}
        labels = decoder_input_batch["labels"]
        score_reinforce = 0
        self.model.is_sample = False
            
        with tf.device('/gpu:0'):       

            output_state_aw = self.model.decoder_per_feed_previous(encoder_input_batch, 
                                            decoder_input_batch,
                                            self.encoder_state,
                                            self.encoder_outputs, 
                                            encoder_input_batch["seq_length"],
                                            self.query_state, feed_previous, is_prev_decoder=True)


            prev_output_switch = output_state_aw[-1][1]["output_switch"]
            output_state_aw = output_state_aw[:-1] 

            max_prop_steps = int(self.model.args["Decoder"]["max_prop_steps"]) -1 
            decoder_outputs = output_state_aw[-3 * max_prop_steps : -2 * max_prop_steps]
            prev_decoder_logits = output_state_aw[:max_prop_steps]
            prev_attention_weights = output_state_aw[-4 * max_prop_steps : -3 * max_prop_steps]
            prev_decoder_tokens = tf.cast(tf.argmax(tf.stack(prev_decoder_logits,axis=1),axis=2) , tf.int32)
            prev_decoder_embeddings  = tf.nn.embedding_lookup(self.model.word_embeddings, prev_decoder_tokens)

            final_embedding =  prev_decoder_embeddings
            final_tokens    = prev_decoder_tokens

            
            final_embedding,  masked_weights_props, prev_decoder_mean_attn, total_encoder_teacher_attention_weights, question_gate, question_gate_policy  = self.post_first_decoder( 
                                                                                                                                                              final_tokens, prev_attention_weights, self.query_state)
            
            print("Prev decoder mean attn", prev_decoder_mean_attn, total_encoder_teacher_attention_weights)
            if self.model.args["Question_encoder"]["use_gate_policy"] == "True" and self.model.args["Question_encoder"]["use_rl_on_gate"] == "True":
                score_prev_decoder = tf.py_func(compute_bleu, (prev_decoder_tokens, labels), [tf.float32], name="crossent_bleu") 
                score_prev_decoder = tf.concat(score_prev_decoder,axis=0)
                question_gate_rewards = compute_question_gate_rewards(score_prev_decoder, question_gate)
                question_gate_loss = bandits_reinforce(question_gate_rewards,question_gate,question_gate_policy)
            
        
        with tf.device('/gpu:0'):       

            print ("final embedding",final_embedding)
            output_state_aw = self.model.decoder_per_feed_previous(encoder_input_batch,
                                            decoder_input_batch,
                                            self.encoder_state,
                                            self.encoder_outputs,
                                            encoder_input_batch["seq_length"],
                                            self.query_state, feed_previous, is_prev_decoder=False, 
                                            prev_decoder_state=final_embedding, prop_indices=final_tokens,
                                            prev_coverage_vec_different = total_encoder_teacher_attention_weights,
                                            masked_weights_props = masked_weights_props, 
                                            prev_decoder_mean_attn =  prev_decoder_mean_attn, 
                                            question_gate = question_gate)


            final_outputs["initial_states"] = output_state_aw[-1][0]
            self.initial_states = final_outputs["initial_states"]
            final_outputs["new_state_values"] = output_state_aw[-1][1]
            print ("NEWSTATEVALUES", final_outputs["new_state_values"])
            output_state_aw = output_state_aw[:-1]

            final_outputs["attention_weights_prop"] = final_outputs["new_state_values"]["attention_values_prop"]  ##

            final_outputs["outputs"] = output_state_aw[:max_steps]
            
            predicted_tokens_reinforce = tf.cast(tf.argmax(tf.stack(final_outputs["outputs"],axis=1),axis=2) , tf.int32)
            #predicted_tokens_reinforce = tf.to_int32(tf.stack(predicted_tokens_reinforce, axis=1))
            final_outputs["prev_decoder_outputs"] = prev_decoder_logits
            final_outputs["prev_attention_weights"] = prev_attention_weights  ##
            final_outputs["prev_decoder_tokens"] = tf.unstack(prev_decoder_tokens,axis=1)
            final_outputs["state"] = output_state_aw[max_steps]
            final_outputs["attention_weights"] = output_state_aw[-4* max_steps: -3 * max_steps]  ##
            final_outputs["pos_outputs"] = output_state_aw[-3 * max_steps : -2 * max_steps]
            final_outputs["predictions"] = output_state_aw[-2*max_steps:-1*max_steps]
            final_outputs["question_logits"] = self.question_logits
            final_outputs["property_attn_values"] = output_state_aw[-1*max_steps:]
            final_outputs["beta_values"] = output_state_aw[-1*max_steps:]
            final_outputs["property_attention_weights"] = self.prop
            final_outputs["output_switch"] = final_outputs["new_state_values"]["output_switch"]  ##
            final_outputs["prev_output_switch"] = prev_output_switch   
            final_outputs["question_gates"] = question_gate
            
            if self.model.args["Question_encoder"]["use_rl_on_gate"] == "True":
                final_outputs["question_gate_loss"] = question_gate_loss
                final_outputs["question_gate_rewards"] = question_gate_rewards
            else:
                final_outputs["question_gate_loss"] = tf.zeros([1])
                final_outputs["question_gate_rewards"] = tf.zeros([1])
                    ##
            
            if (self.model.args["Hyperparams"]["use_only_rl"] == "True" or self.model.args["Hyperparams"]["rl_plus_crossent"] == "True") or self.model.args["Question_encoder"]["use_gate_policy"] == "True":

                with tf.device('/gpu:0'):       
                    self.model.is_sample = False
        
                    
                    is_qbleu = self.model.args["Hyperparams"]["use_qbleu"] == "True"
                
                    if is_qbleu:
                        score_reinforce  = tf.py_func(answerability_score.reinforce_compute_score_parallel, (predicted_tokens_reinforce, labels),[tf.float32], name="reinforce_dbleu")
                    else:
                        score_reinforce  = tf.py_func(compute_bleu, (predicted_tokens_reinforce, labels), [tf.float32], name="reinforce_bleu")
        
                    predicted_tokens_crossent  = prev_decoder_tokens 
                    print ("Predicted_cross_en_size", predicted_tokens_crossent.get_shape(), labels.get_shape())    
                    
                    if self.model.args["Hyperparams"]["use_qbleu"] == "True":
                        score_crossent = tf.py_func(answerability_score.reinforce_compute_score_parallel, (predicted_tokens_crossent, labels), 
                                                    [tf.float32], name="crossent_qbleu")
                    
                    else:
                        score_crossent = tf.py_func(compute_bleu, (predicted_tokens_crossent, labels), [tf.float32], name="crossent_bleu")

                    final_outputs["bleu_reinforce"] =score_reinforce 
                    final_outputs["bleu_crossent"] = score_crossent

                    
            if self.model.args["Question_encoder"]["use_gate_policy"] == "True":
                score_reinforce = tf.stack(score_reinforce, axis=0)
                score_crossent = tf.stack(score_crossent, axis=0)
                question_gate_rewards = (score_reinforce - score_crossent)
                # question_gate_rewards = compute_question_gate_rewards(score_crossent, question_gate)
                question_gate_loss = bandits_reinforce(question_gate_rewards,question_gate,question_gate_policy)
                final_outputs["question_gate_loss"] = question_gate_loss
                final_outputs["question_gate_rewards"] = question_gate_rewards
    

        for key in final_outputs:
            print("DEBUG {} : {}".format(key, final_outputs[key]))
        return final_outputs
 
