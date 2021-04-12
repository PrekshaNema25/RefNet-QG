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

def _get_ngrams_with_counter(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in xrange(1, max_order + 1):
    for i in xrange(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def _get_ngrams(n, text):
  """Calculates n-grams.
  Args:
    n: which n-grams to calculate
    text: An array of tokens
  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set

def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.
  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []
  bleu_scores = []
  for (references_temp, translations_temp) in zip(reference_corpus, translation_corpus):
       ref_temp = [np.trim_zeros(references_temp.flatten())]
       trans_temp = [np.trim_zeros(translations_temp.flatten())]

       for (references, translations) in zip(ref_temp, trans_temp):
          reference_length += len(references)
          translation_length += len(translations)
          ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
          translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

          overlap = dict((ngram,
                          min(count, translation_ngram_counts[ngram]))
                         for ngram, count in ref_ngram_counts.items())

          for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
          for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]

          precisions = [0] * max_order
          smooth = 1.0

          for i in xrange(0, max_order):
            if possible_matches_by_order[i] > 0:
              precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
              if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                    i]
              else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
            else:
              precisions[i] = 0.0

          if max(precisions) > 0:
            p_log_sum = sum(math.log(p) for p in precisions if p)
            geo_mean = math.exp(p_log_sum / max_order)

          if use_bp:
            ratio = translation_length / (reference_length + 0.001)
            bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
          bleu = geo_mean * bp
          bleu_scores.append(bleu)
  return np.float32(bleu_scores)


class BasicSeq2SeqWithAttentionwithQuery:

    def __init__(self, args, embedding_initializer=None, vocab_length=None, vocab_char_length = None, vocab_dict=None, dropout_param=None):
        self.args = args
        self.is_sample = False
        self.dropout_param=dropout_param

        print ("Dropout_Param", dropout_param)
        # for BLEU and QBLEU computation
        self.vocab_dict = vocab_dict

        with tf.variable_scope("seq2seq_basic_variables", reuse=tf.AUTO_REUSE):

            if self.args["Decoder"]["use_diff_out_proj"] == "True":  
                self.prev_output_projection = [tf.get_variable("prev_output_proj_w", shape=[self.args["Decoder"]["hidden_size"],
                                                                              vocab_length], dtype=tf.float32),
                                               tf.get_variable("prev_output_bias_w", shape=vocab_length, dtype=tf.float32)]

            self.output_projection = [tf.get_variable("output_proj_w", shape=[self.args["Decoder"]["hidden_size"],
                                                                              vocab_length], dtype=tf.float32),
                                      tf.get_variable("output_bias_w", shape=vocab_length, dtype=tf.float32)]


            self.vocab_length = vocab_length

            if "trainable" in args["Embedding"]:
                trainable = bool(args["Embedding"]["trainable"] == "True")

            else:
                trainable = False

            self.word_embeddings = self.word_embedding_layer(embedding_initializer = embedding_initializer,embedding_trainable=trainable, var_name = "word_embeddings",vocab_length=None,
                                                             embedding_size=int(args["Embedding"]["word_embedding_size"]))

            if self.args["Embedding"]["use_char_embed"] == "True":
                self.char_init_embedding_layer(embedding_trainable=True, var_name="char_embeddings", num_chars=int(vocab_char_length), 
                                                                  char_embedding_size=int(args["Embedding"]["char_in_embed_size"]))

            if self.args["Embedding"]["use_positional_embeddings"] == "True":
                self.positional_embedding_layer = tf.get_variable("positional_embeds", shape=[2, int(self.args["Embedding"]["position_embeddings_dims"])])

        
    def fully_connected(self,x,dim_size,activation_fn=None,keep_prob=1):
        is_train = self.is_train
        fc1 = tf.nn.dropout(tf.contrib.layers.fully_connected(x, dim_size, activation_fn=activation_fn), keep_prob=keep_prob)
        #if self.args["Hyperparams"]["use_batch_norm"] == "True" and keep_prob != 1:
        #    fc1 = tf.layers.batch_normalization(fc1, training=is_train)
        return fc1

    def _multi_conv1d(self, in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
        with tf.variable_scope(scope or "multi_conv1d"):
            assert len(filter_sizes) == len(heights)
            outs = []
            for filter_size, height in zip(filter_sizes, heights):
                if filter_size == 0:
                    continue
                print ("DEBUG: in_ size {}  ".format(in_.get_shape()))
                #out = tf.nn.conv1d(in_, filters=[filter_size,  20, height], stride=2, padding="VALID")
                out = tf.layers.conv1d(in_, filters = int(filter_size), kernel_size=height, reuse=tf.AUTO_REUSE)
                print ("DEBUG: out size {}".format(out))
                out = tf.reduce_max(out, axis=1)
                print ("DEBUG: out size {}".format(out))
                outs.append(out)
            concat_out = tf.concat(outs, axis=-1)
        return concat_out

    def _extract_argmax_and_embed(self, update_embedding=True):
        """Get a loop_function that extracts the previous symbol and embeds it.
          Args:
              update_embedding: Boolean; if False, the gradients will not propagate
                                through the embeddings.
            Returns:
                A loop function.
        """
        def loop_function(prev, _):
            prev_symbol_sample = tf.distributions.Categorical(probs=prev).sample()
            prev_symbol_greedy = tf.argmax(prev, 1)

            return prev_symbol_sample, prev_symbol_greedy
        return loop_function
    
    def select_cell(self,  var_name, cell_name, hidden_size, num_layers, dropout_param):
        """
        Args:
            args: A dictionary with arguments.
            cell_name: Name of the cell class.
        Returns:
            cell_class: Given the type of RNN Cell to be used
            as a basic unit, corresponding class is returned.
        """
        print ("DEBUG: hidden size is  {} {}". format(var_name, hidden_size))
        with tf.variable_scope("RNNCells_{}".format(var_name)):
            if cell_name=='GRU':
                list_of_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size), output_keep_prob=dropout_param, input_keep_prob=dropout_param) for _ in range(num_layers)]
            elif cell_name=="LNLSTM":
                list_of_cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size) for _ in range(num_layers)]
            else:
                list_of_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hidden_size), output_keep_prob=dropout_param, input_keep_prob=dropout_param) for _ in range(num_layers)]

        return list_of_cells

    def word_embedding_layer(self, embedding_initializer=None,
                             embedding_trainable=True,
                             var_name = "",
                             vocab_length=None,
                             embedding_size=None):
        """
        Args:
            embedding_initializer: Initial matrix, (glove vectors)
            to be used to initialize the embedding matrix.
            embedding_trainable: To backpropogate through embedding matrix
            vocab_length: Total number of words in vocab. It will not be 
            used if initial matrix is given.
            embedding_size: If initial matrix is not given, embedding_size
            is needed to initialize the embedding variable
        Returns:
            A class variable is initialized with the embedding layer
        """
        with tf.variable_scope("WordEmbeddingLayer"):
            if embedding_initializer is not None:
                embeddings = tf.get_variable(name="word_embedding_{}".format(var_name),
                                             initializer=embedding_initializer,
                                             trainable=embedding_trainable,
                                             dtype=tf.float32)
            else:
                print("DEBUG: Embedding Initializer not passed")
                print ("DEBUG: Vocab_length : {}".format(self.vocab_length))
                embeddings = tf.get_variable(name="word_embedding_{}".format(var_name),
                                             shape=[self.vocab_length, embedding_size],
                                             trainable=embedding_trainable,
                                             initializer=None,
                                             dtype=tf.float32)
        return embeddings

    def char_init_embedding_layer(self, embedding_trainable=True,
                             var_name="",
                             num_chars=68, 
                             char_embedding_size=50):
        with tf.variable_scope("CharEmbeddingLayer", reuse=tf.AUTO_REUSE):
            char_emb_mat = tf.get_variable("char_embedding_{}".format(var_name),
                                           shape=[num_chars, char_embedding_size],
                                           dtype=tf.float32)

        self.char_embeddings = char_emb_mat

    def char_embedding_layer_compute(self, encoder_input, word_length):
        with tf.variable_scope("char"):
            temp_char_embeddings = tf.nn.embedding_lookup(self.char_embeddings,  encoder_input)
            temp_char_embeddings = tf.reshape(temp_char_embeddings,
                                              [-1, int(self.args["Embedding"]["max_word_length"]), int(self.args["Embedding"]["char_in_embed_size"])])
            

            filter_sizes = list(map(float, self.args["Embedding"]["char_out_channel_dims"].split(",")))
            heights = list(map(int, self.args["Embedding"]["char_filter_heights"].split(",")))

        with tf.variable_scope("char_conv", reuse=tf.AUTO_REUSE):
            temp_char_embeddings = self._multi_conv1d(temp_char_embeddings,
                                                     filter_sizes,
                                                     heights,
                                                     "VALID",
                                                     True,
                                                     self.dropout_param)
            temp_char_embeddings = tf.reshape(temp_char_embeddings,[-1, word_length, int(self.args["Embedding"]["char_out_embed_size"])])
        
        return temp_char_embeddings

    def compute_all_embeddings(self,  input_words, input_chars):

        print ("DEBUG: Input words", input_words, self.word_embeddings)
      
        embedded_words = tf.nn.embedding_lookup(self.word_embeddings, input_words)
        
        if self.args["Embedding"]["use_char_embed"] == "True":
            print ("DEBUG: Word length ", input_words.get_shape())
            embedded_chars = self.char_embedding_layer_compute(input_chars, word_length = input_words.get_shape()[-1].value)
            concat_word_char_embed = tf.concat([embedded_words, embedded_chars], axis=-1)
            return concat_word_char_embed
        else:
            return embedded_words
        
    def passage_encoder(self, encoder_input_batch):
        """
        Args:
            encoder_input_batch: A dictionary with required input data
            args: A list of hyper parameters.
        Returns
        """
        input_words = encoder_input_batch["word"]
        input_chars = encoder_input_batch["char"]


        print ("DEBUG: Size for input words {} and Size for input chars {}".format(input_words, input_chars))

        self.encoder_fw_cells = self.select_cell("encoder_fw", self.args["Encoder"]["cell_type"], int(self.args["Encoder"]["hidden_size"]), int(self.args["Encoder"]["num_layers"]), dropout_param=self.dropout_param)
        self.encoder_bw_cells = self.select_cell("encoder_bw",self.args["Encoder"]["cell_type"], int(self.args["Encoder"]["hidden_size"]), int(self.args["Encoder"]["num_layers"]), dropout_param=self.dropout_param)

        

        concat_word_char_embed = self.compute_all_embeddings( input_words, input_chars)
        embed_size = concat_word_char_embed.get_shape()[-1].value

        if self.args["Embedding"]["use_positional_embeddings"] == 'True':
          print ("PASSAGE ENCODER", encoder_input_batch["positional"].get_shape())
          concat_word_char_embed = tf.concat([concat_word_char_embed,  encoder_input_batch["positional"]],axis=-1)

        if "only_gcn" in self.args["Encoder"] and self.args["Encoder"]["only_gcn"] == "True":
           encoder_outputs = concat_word_char_embed
           encoder_state = tf.reduce_mean(encoder_outputs, 1)
           return encoder_outputs, encoder_state  
      
        print ("PASSAGE ENCODER Embedding layer output {}".format(concat_word_char_embed.get_shape()))
        temp_outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                       self.encoder_fw_cells, 
                       self.encoder_bw_cells,
                       concat_word_char_embed,
                       sequence_length= encoder_input_batch["seq_length"], dtype=tf.float32)
        
        encoder_outputs, encoder_fw_state, encoder_bw_state = temp_outputs
        
        encoder_fw_state = encoder_fw_state[-1]
        encoder_bw_state = encoder_bw_state[-1]
        encoder_state = tf.concat([encoder_fw_state[-1], encoder_bw_state[-1]], axis=-1)
        
        print ("DEBUG: Encoder_Outputs and Encoder State size {}, {}".format(encoder_outputs.get_shape(), encoder_state.get_shape()))
        return encoder_outputs, encoder_state

    def question_encoder(self, question_words_embed,question_seq_length):

        """
        Args:
            encoder_input_batch: A dictionary with required input data
            args: A list of hyper parameters.
        Returns
        """

        self.question_encoder_fw_cells = self.select_cell("question_encoder_fw", self.args["Question_encoder"]["cell_type"], int(self.args["Question_encoder"]["hidden_size"]), int(self.args["Question_encoder"]["num_layers"]), dropout_param=self.dropout_param)
        self.question_encoder_bw_cells = self.select_cell("question_encoder_bw",self.args["Question_encoder"]["cell_type"], int(self.args["Question_encoder"]["hidden_size"]), int(self.args["Question_encoder"]["num_layers"]), dropout_param=self.dropout_param)

        print ("Question ENCODER Embedding layer output {}".format(question_words_embed.get_shape()))
        temp_outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                       self.question_encoder_fw_cells,
                       self.question_encoder_bw_cells,
                       question_words_embed,
                       sequence_length=question_seq_length, dtype=tf.float32)

        encoder_outputs, encoder_fw_state, encoder_bw_state = temp_outputs
        
        encoder_fw_state = encoder_fw_state[-1]
        encoder_bw_state = encoder_bw_state[-1]
        encoder_state = tf.concat([encoder_fw_state[-1], encoder_bw_state[-1]], axis=-1)
        
        print ("DEBUG: Question Encoder_Outputs and Encoder State size {}, {}".format(encoder_outputs.get_shape(), encoder_state.get_shape()))
        return encoder_outputs, encoder_state
     
    def gate_policy(self,question_state):

        dim_size = question_state.get_shape()[-1].value
        fc1 = self.fully_connected(question_state, dim_size, activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
        logits = tf.squeeze(self.fully_connected(fc1, 1, activation_fn=None),axis=1)
        probs = tf.nn.sigmoid(logits)
        action_sampled = tf.distributions.Bernoulli(probs=probs).sample()
        return action_sampled, logits

    def gate_soft(self,question_state):

        dim_size = 2 * int(self.args["Encoder"]["hidden_size"])#question_state.get_shape()[-1].value
        fc1 = self.fully_connected(question_state, dim_size, activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
        gate = tf.squeeze(self.fully_connected(fc1, 1, activation_fn=tf.nn.sigmoid),axis=1)
        return gate

    def query_encoder(self, query_input_batch, temp_query_embeds=None):

        query_input_words = query_input_batch["word"]
        query_input_chars = query_input_batch["char"]

        self.query_fw_cells = self.select_cell( "query_fw", self.args["Query"]["cell_type"], int(self.args["Query"]["hidden_size"]), int(self.args["Query"]["num_layers"]), dropout_param=self.dropout_param)
        self.query_bw_cells = self.select_cell( "query_bw", self.args["Query"]["cell_type"], int(self.args["Query"]["hidden_size"]), int(self.args["Query"]["num_layers"]), dropout_param=self.dropout_param)
        
        concat_word_char_embed = self.compute_all_embeddings(query_input_words, query_input_chars)
        if self.args["Query"]["use_position"] == "True":
           concat_word_char_embed = tf.concat([concat_word_char_embed, temp_query_embeds], axis=-1)


        with tf.variable_scope("query_encoder"):
            temp_outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                           self.query_fw_cells, 
                           self.query_bw_cells,
                           concat_word_char_embed,
                           sequence_length= query_input_batch["seq_length"], dtype=tf.float32)
            
        query_outputs, query_fw_state, query_bw_state = temp_outputs
        
        query_fw_state = query_fw_state[-1]
        query_bw_state = query_bw_state[-1]
        print ("DEBUG: query_fw_states, query_bw_states {}, {}".format(query_fw_state[-1].get_shape(), query_bw_state[-1].get_shape()))
        query_state = tf.concat([query_fw_state[-1], query_bw_state[-1]], axis=-1)
	
        print ("DEBUG: Query_Outputs and Query State size {}, {}".format(query_outputs.get_shape(), query_state.get_shape()))
        return query_outputs, query_state
    
    def attention_compute_prev_code_1(self, var_name, memory_states, masked_weights, query, prev_coverage=None, property_attn_weights=None, prev_decoder_mean_attn= None): 

        num_memory_st = memory_states.get_shape()[1].value
        dim_size = memory_states.get_shape()[2].value

        if prev_coverage is not None:
           prev_coverage = tf.expand_dims(prev_coverage, axis=2)
           
        if property_attn_weights is not None:
           property_attn_weights = tf.stack(property_attn_weights, axis=2)

        with tf.variable_scope("attention_proj{}".format(var_name), reuse=tf.AUTO_REUSE):
             memory_state_new = self.fully_connected(memory_states,dim_size,activation_fn=None, keep_prob=self.dropout_param)
             v = tf.get_variable("attntion_v_vec", [dim_size])
             if prev_coverage is not None:
                with tf.variable_scope("attention_proj_prev_cov", reuse=tf.AUTO_REUSE):
                    prev_coverage_new =  self.fully_connected(prev_coverage,dim_size,activation_fn=None, keep_prob = self.dropout_param)

             if property_attn_weights is not None:
                with tf.variable_scope("attention_proj_prop_dep", reuse=tf.AUTO_REUSE):
                    property_attn_weights_new =  self.fully_connected(property_attn_weights, dim_size, activation_fn=None, keep_prob=self.dropout_param)

        with tf.variable_scope("Attention_{}".format(var_name), reuse=tf.AUTO_REUSE):
            y = self.fully_connected(query, dim_size, activation_fn=None, keep_prob=self.dropout_param)
            y = tf.expand_dims(y, axis=[1])
            temp_sum = memory_state_new + y
            if prev_coverage is not None:
                temp_sum += prev_coverage_new
            
            if property_attn_weights is not None:
                temp_sum += property_attn_weights_new 
            
            s = tf.reduce_sum(v* tf.nn.tanh(temp_sum), 2) 
            if masked_weights is not None:
            	s = tf.where(tf.cast(masked_weights, tf.bool), s,-10000*tf.ones_like(s))   
            if prev_decoder_mean_attn is not None:
                print (prev_decoder_mean_attn,  s)
                s = s* (1-prev_decoder_mean_attn) 
            a = tf.nn.softmax(s)
            if masked_weights is not None:
                a = tf.where(tf.cast(masked_weights, tf.bool), a, 0.00001*tf.ones_like(a))
            d = tf.reduce_sum(tf.expand_dims(a, axis=2) * memory_states, 1)

        return s, a
    

    def gated_self_attention(self, var_name, memory_states, masked_weights):

        dimension = memory_states.get_shape()[-1].value

        with tf.variable_scope("Self_Attention_Scores_{}".format(var_name), reuse=tf.AUTO_REUSE):
           memory_states_projected = self.fully_connected(memory_states, dimension, activation_fn=None)

           attn_weights = tf.matmul(memory_states_projected, tf.transpose(memory_states,[0,2,1]))
           
           new_masked_weights = tf.expand_dims(masked_weights, -1)
           new_masked_weights = tf.tile(new_masked_weights, multiples=[1,1,masked_weights.get_shape()[1].value])
           new_attn_weights = tf.nn.softmax(tf.where(tf.cast(new_masked_weights, tf.bool), attn_weights, -10000*tf.ones_like(attn_weights)),axis=-1)

           attn_repr = tf.matmul(new_attn_weights,memory_states)


        with tf.variable_scope("Self_Attention_Gates_{}".format(var_name), reuse=tf.AUTO_REUSE):
           new_attn_repr = self.fully_connected(tf.concat([attn_repr,memory_states],axis=-1), dimension, activation_fn=tf.nn.tanh)
           attn_gate = self.fully_connected(tf.concat([attn_repr,memory_states],axis=-1), dimension, activation_fn=tf.nn.sigmoid)
           norm = tf.norm(attn_gate)
           tf.summary.scalar('Self Attention Gate (Norm)',norm)
           final_repr = tf.multiply(attn_gate,new_attn_repr) + tf.multiply( (1-attn_gate) ,memory_states)  

        return final_repr

    def gated_three_level_self_attention_bahdanau(self, var_name, memory_states, coattention_memory_states, masked_weights):

        dimension = memory_states.get_shape()[-1].value
        max_time = memory_states.get_shape()[1].value
        with tf.variable_scope("Self_Attention_Scores_{}".format(var_name), reuse=tf.AUTO_REUSE):
            
           memory_states_tile1 = tf.tile(tf.expand_dims(memory_states,1), [1,max_time,1,1])
           memory_states_tile2 = tf.tile(tf.expand_dims(memory_states,2), [1, 1,max_time,1])
           memory_matrix = tf.concat([memory_states_tile1,memory_states_tile1],axis=-1)
           memory_matrix_projected =  self.fully_connected(memory_matrix, dimension, activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
           memory_matrix_final = tf.squeeze(self.fully_connected(memory_matrix_projected, 1, activation_fn=None),-1)     

           new_masked_weights = tf.expand_dims(masked_weights, -1)
           new_masked_weights = tf.tile(new_masked_weights, multiples=[1,1,masked_weights.get_shape()[1].value])
           new_attn_weights = tf.nn.softmax(tf.where(tf.cast(new_masked_weights, tf.bool), memory_matrix_final, -10000*tf.ones_like(memory_matrix_final)),axis=-1)

           attn_repr = tf.matmul(new_attn_weights,memory_states)

        with tf.variable_scope("Gate_between_encoder_coattn_{}".format(var_name), reuse=tf.AUTO_REUSE):
           comb_enc_gate = self.fully_connected(tf.concat([coattention_memory_states, memory_states],axis=-1), dimension, activation_fn=tf.nn.sigmoid)
           comb_enc_repr = tf.multiply(comb_enc_gate, memory_states) + tf.multiply((1-comb_enc_gate),coattention_memory_states)

        with tf.variable_scope("Self_Attention_Gates_{}".format(var_name), reuse=tf.AUTO_REUSE):
           attn_gate = self.fully_connected(tf.concat([attn_repr,comb_enc_repr],axis=-1), dimension, activation_fn=tf.nn.sigmoid)
           norm = tf.norm(attn_gate)
           tf.summary.scalar('Self Attention Gate (Norm)',norm)
           final_repr = tf.multiply(attn_gate,attn_repr) + tf.multiply((1-attn_gate), comb_enc_repr)  

        return final_repr

    def gated_three_level_self_attention(self, var_name, memory_states, coattention_memory_states, masked_weights):

        dimension = memory_states.get_shape()[-1].value

        with tf.variable_scope("Self_Attention_Scores_{}".format(var_name), reuse=tf.AUTO_REUSE):
           memory_states_projected = self.fully_connected(memory_states, dimension, activation_fn=None, keep_prob=self.dropout_param)
           attn_weights = tf.matmul(memory_states_projected, tf.transpose(memory_states,[0,2,1]))
           
           new_masked_weights = tf.expand_dims(masked_weights, -1)
           new_masked_weights = tf.tile(new_masked_weights, multiples=[1,1,masked_weights.get_shape()[1].value])
           new_attn_weights = tf.nn.softmax(tf.where(tf.cast(new_masked_weights, tf.bool), attn_weights, -10000*tf.ones_like(attn_weights)),axis=-1)

           attn_repr = tf.nn.dropout(tf.matmul(new_attn_weights,memory_states, keep_prob=self.dropout_param))

        with tf.variable_scope("Gate_between_encoder_coattn_{}".format(var_name), reuse=tf.AUTO_REUSE):
           #new_comb_repr = self.fully_connected(tf.concat([coattention_memory_states, memory_states],axis=-1), dimension, activation_fn=tf.nn.tanh)
           comb_enc_gate = self.fully_connected(tf.concat([coattention_memory_states, memory_states],axis=-1), dimension, activation_fn=tf.nn.sigmoid)
           comb_enc_repr = tf.nn.dropout(tf.multiply(comb_enc_gate, memory_states) + tf.multiply((1-comb_enc_gate),coattention_memory_states), keep_prob=self.dropout_param)

           norm1 = tf.norm(comb_enc_gate)
           tf.summary.scalar('Self Attention Comb enc Gate (Norm)',norm1)

        with tf.variable_scope("Self_Attention_Gates_{}".format(var_name), reuse=tf.AUTO_REUSE):
           #new_attn_repr = self.fully_connected(tf.concat([attn_repr,comb_enc_repr],axis=-1), dimension, activation_fn=tf.nn.tanh)
           attn_gate = self.fully_connected(tf.concat([attn_repr,comb_enc_repr],axis=-1), dimension, activation_fn=tf.nn.sigmoid)
           norm = tf.norm(attn_gate)
           tf.summary.scalar('Self Attention Gate (Norm)',norm)
           final_repr = tf.multiply(attn_gate,attn_repr) + tf.multiply( (1-attn_gate), comb_enc_repr)  

        return final_repr

    def self_attention_lstm(self, var_name, memory_states, masked_weights, seq_len=None):

        dimension = memory_states.get_shape()[-1].value

        with tf.variable_scope("Self_Attention_Scores_{}".format(var_name), reuse=tf.AUTO_REUSE):
           memory_states_projected = self.fully_connected(memory_states, dimension, activation_fn=None, keep_prob=self.dropout_param)
           attn_weights = tf.matmul(memory_states_projected, tf.transpose(memory_states,[0,2,1]))
           
           new_masked_weights = tf.expand_dims(masked_weights, -1)
           new_masked_weights = tf.tile(new_masked_weights, multiples=[1,1,masked_weights.get_shape()[1].value])
           new_attn_weights = tf.nn.softmax(tf.where(tf.cast(new_masked_weights, tf.bool), attn_weights, -10000*tf.ones_like(attn_weights)),axis=-1)
           attn_repr = tf.matmul(new_attn_weights,memory_states)

           self.self_attn_encoder_fw_cells = self.select_cell( "new_encoder_fw", self.args["Encoder"]["cell_type"], int(self.args["Encoder"]["hidden_size"]), 1, dropout_param=self.dropout_param)
           self.self_attn_encoder_bw_cells = self.select_cell( "new_encoder_bw",self.args["Encoder"]["cell_type"], int(self.args["Encoder"]["hidden_size"]), 1, dropout_param=self.dropout_param)

           new_repr = tf.concat([memory_states, attn_repr], axis=-1)

           temp_outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                       self.self_attn_encoder_fw_cells,
                       self.self_attn_encoder_bw_cells,
                       new_repr,
                       sequence_length= seq_len, dtype=tf.float32)

           encoder_outputs, encoder_fw_state, encoder_bw_state = temp_outputs
           encoder_fw_state = encoder_fw_state[-1]
           encoder_bw_state = encoder_bw_state[-1]
           encoder_state = tf.concat([encoder_fw_state[-1], encoder_bw_state[-1]], axis=-1)
        

        return encoder_outputs, encoder_state
   
    def compute_multi_head_attention(self, var_name,  memory_states, masked_weights, query, return_all=False, num_heads=2, property_coverage=False,
                                     prev_coverage_vec=None, property_attention_weights = None, property_query_states=None, prev_decoder_mean_attn= None):

        """
        Args:
        """
        #print ("Mem States, Mem seq length,query",memory_states.get_shape(),memory_sequence_length.get_shape(),query.get_shape() ) 

        alignment_values_per_head = []
        attention_scores_per_head = []
        memory_size = memory_states.get_shape()[-1].value
        prop_cov_vec = tf.zeros((int(self.args["Hyperparams"]["batch_size"]), memory_states.get_shape()[1].value)) 

        print ("Memory State Shape", memory_states.get_shape())
        for i in range(num_heads):
            if property_coverage:
              query = property_query_states[i]
              new_attention_scores,  new_attention_weights = self.attention_compute_prev_code_1(str(i) + "_" + var_name, memory_states,masked_weights , query,
                                          prev_coverage=prop_cov_vec, prev_decoder_mean_attn= prev_decoder_mean_attn)
              prop_cov_vec += new_attention_weights  
              alignment_values_per_head.append(new_attention_weights)
              attention_scores_per_head.append(new_attention_scores)

            elif prev_coverage_vec is not None: 
              new_attention_scores,  new_attention_weights = self.attention_compute_prev_code_1(str(i) + "_" + var_name, memory_states, masked_weights, query,
                                          prev_coverage=prev_coverage_vec, property_attn_weights=property_attention_weights, prev_decoder_mean_attn= prev_decoder_mean_attn)
              alignment_values_per_head.append(new_attention_weights)
              attention_scores_per_head.append(new_attention_scores)

            else:
              new_attention_scores,  new_attention_weights = self.attention_compute_prev_code_1(str(i) + "_" + var_name, memory_states, masked_weights, query, prev_decoder_mean_attn= prev_decoder_mean_attn)
              alignment_values_per_head.append(new_attention_weights)
              attention_scores_per_head.append(new_attention_scores)

        new_memory_states = []
        for alignments in alignment_values_per_head:
            #print ("DEBUG : alignment shape {}, {}, {}".format(len(alignments) , alignments[0].get_shape(), alignments[1].get_shape()))
            temp = tf.tile(tf.expand_dims(alignments, axis=2), multiples=[1,1, memory_states.get_shape()[2].value])
            temp = temp * memory_states
            new_memory_states.append(tf.reduce_sum(temp, axis=1))
        
        if return_all:
            with tf.variable_scope("non_linear_attention_mechanims_{}".format(var_name), reuse=tf.AUTO_REUSE):
                temp_projected = tf.stack(new_memory_states, axis=1)
                temp_projected = self.fully_connected(temp_projected, memory_size, activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
                temp_projected = tf.unstack(temp_projected, axis=1)
            return temp_projected, alignment_values_per_head, attention_scores_per_head
        else:
            with tf.variable_scope("attention_comb_layer_{}".format(var_name), reuse=tf.AUTO_REUSE):
                if (num_heads == 1):
                   new_context = new_memory_states
                else:
                   new_memory_states = tf.concat(new_memory_states, axis=-1)
                   new_context = self.fully_connected(new_memory_states, memory_size, activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
            return new_context, alignment_values_per_head, attention_scores_per_head
                 
    def generator_switch(self,  input_to_generator):
        with tf.variable_scope("generator_switch", reuse=tf.AUTO_REUSE):
            output_switch = self.fully_connected(input_to_generator, 1, activation_fn=tf.nn.sigmoid)
        return output_switch

    def generator_switch_prop_3gates(self,  input_to_generator):
        dim_size = int(self.args["Decoder"]["hidden_size"])
        with tf.variable_scope("generator_switch_prop_content", reuse=tf.AUTO_REUSE):
            input_to_generator = self.fully_connected(input_to_generator, dim_size, activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
        
        with tf.variable_scope("generator_switch_full", reuse=tf.AUTO_REUSE):
            output_switch_full = tf.nn.softmax(self.fully_connected(input_to_generator, 3, activation_fn=None), axis=-1)
        
        return output_switch_full

    def generator_switch_prop(self,  input_to_generator):
        with tf.variable_scope("generator_switch_prop_content", reuse=tf.AUTO_REUSE):
            output_switch_prop_cont = self.fully_connected(input_to_generator, 1, activation_fn=tf.nn.sigmoid)
        
        with tf.variable_scope("generator_switch_full", reuse=tf.AUTO_REUSE):
            output_switch_full = self.fully_connected(input_to_generator, 1, activation_fn=tf.nn.sigmoid)
        
        return output_switch_full, output_switch_prop_cont

    def _get_the_comb_projection(self, vocab_projection, attention_values, output_switch, content_weights,  word_indices=None, is_inference=False):

        batch_size = int(self.args["Hyperparams"]["batch_size"])
        if is_inference:
            batch_size = int(self.args["Hyperparams"]["batch_size"])*int(self.args["BeamSearch"]["beam_size"])

        seq_length = attention_values[0].get_shape()[-1].value
        new_vocab_projection = output_switch *vocab_projection
        new_attention_projection = (1 - output_switch) * (attention_values[0] * content_weights)

        batch_nums = tf.range(0, batch_size)
        batch_nums = tf.tile(tf.expand_dims(batch_nums, axis=1), [1, seq_length])
        indices = tf.stack([batch_nums, word_indices], axis=2)
        new_attention_projection_proj = tf.scatter_nd(indices, new_attention_projection, shape=(int(batch_size), self.vocab_length))
        
        comb_projection = new_vocab_projection + new_attention_projection_proj

        return comb_projection, new_attention_projection


    def _get_the_comb_projection_property_3gates(self, vocab_projection, attention_values, attention_values_different,  output_switch_prop_cont, 
                                                 content_weights,word_indices=None, is_inference = False):

        batch_size = int(self.args["Hyperparams"]["batch_size"])
        if is_inference:
            batch_size = int(self.args["Hyperparams"]["batch_size"])*int(self.args["BeamSearch"]["beam_size"])

        seq_length = attention_values[0].get_shape()[-1].value
        print ("In three gates function\n")
        

        vocab_switch, copy_switch, copy_diff_switch = [tf.expand_dims(i,axis=1) for i in tf.unstack(output_switch_prop_cont, axis=-1) ]

        new_vocab_projection = vocab_switch * vocab_projection
        copy_projection = copy_switch * (attention_values[0] * content_weights)
        copy_different_projection =  copy_diff_switch * (attention_values_different[0] * content_weights)

        batch_nums = tf.range(0, batch_size)
        batch_nums = tf.tile(tf.expand_dims(batch_nums, axis=1), [1, seq_length])
        indices = tf.stack([batch_nums, word_indices], axis=2)
        copy_projection_proj = tf.scatter_nd(indices, copy_projection, shape=(int(batch_size), self.vocab_length))
        
        copy_different_projection_proj = tf.scatter_nd(indices, copy_different_projection, shape=(int(batch_size), self.vocab_length))
        
        comb_projection = new_vocab_projection + copy_projection_proj + copy_different_projection_proj
        return comb_projection, copy_projection + copy_different_projection

    def _init_decoder_params(self, distract_shape=None, is_prev_state=None, var_name=""):
        initial_states = {}

        with tf.variable_scope("Decoder_Copy", reuse = tf.AUTO_REUSE):
          with tf.variable_scope("Init_Decoder_State{}".format(var_name), reuse=tf.AUTO_REUSE):
                  encoder_state_reshaped = self.fully_connected(self.encoder_state, int(self.args["Decoder"]["hidden_size"]), activation_fn=None, keep_prob=self.dropout_param)
                  decoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state_reshaped, encoder_state_reshaped)
                  initial_states["decoder_state_h"] = decoder_state.h
                  initial_states["decoder_state_c"] = decoder_state.c

                  encoder_pos_state_reshaped = self.fully_connected(self.encoder_state, int(int(self.args["Decoder"]["hidden_size"])/4), activation_fn=None, keep_prob=self.dropout_param)
                  decoder_pos_state = tf.contrib.rnn.LSTMStateTuple(encoder_pos_state_reshaped, encoder_pos_state_reshaped)
                  initial_states["decoder_pos_state"] = decoder_pos_state
                  
          attns_state = tf.zeros(shape=(int(self.args["Hyperparams"]["batch_size"]), distract_shape))
          attns_state_different = tf.zeros(shape=(int(self.args["Hyperparams"]["batch_size"]), distract_shape))
          initial_states["attns_state"] = attns_state
          prev_coverage = tf.zeros(shape=(int(self.args["Hyperparams"]["batch_size"]), int(self.args["Encoder"]["max_sequence_length"])))
          prev_coverage_between_decoders = tf.zeros(shape=(int(self.args["Hyperparams"]["batch_size"]), int(self.args["Decoder"]["max_prop_steps"])-1))
          initial_states["prev_coverage"] = prev_coverage
          initial_states["prev_coverage_between_decoders"] = prev_coverage_between_decoders

          batch_size = int(self.args["Hyperparams"]["batch_size"])
          attns_state_different = attns_state

          if self.args["Question_encoder"]["use_question_encoder"] == "True": 
            prev_decoder_attn_state = tf.zeros((batch_size, 2*int(self.args["Question_encoder"]["hidden_size"])), dtype=tf.float32)
            combined_attns_state = tf.zeros((batch_size, 4*int(self.args["Encoder"]["hidden_size"])), dtype=tf.float32)

            if self.args["Question_encoder"]["use_gated_context"] == "True":
               combined_attns_state = tf.zeros((batch_size, 2*int(self.args["Encoder"]["hidden_size"])), dtype=tf.float32)

            if self.args["Question_encoder"]["word_concat"] == "True":
                prev_decoder_attn_state = tf.zeros((batch_size,
                                          2*int(self.args["Question_encoder"]["hidden_size"]) + int(self.args["Embedding"]["word_embedding_size"])), 
                                          dtype=tf.float32)
            ####   Initial States when Question_encoder = False
          elif self.args["Question_encoder"]["use_gated_context"] == "True":
            prev_decoder_attn_state = tf.zeros((batch_size, int(self.args["Embedding"]["word_embedding_size"])), dtype=tf.float32)
            combined_attns_state = tf.zeros((batch_size, 2*int(self.args["Encoder"]["hidden_size"])), dtype=tf.float32)
              
          else:
            prev_decoder_attn_state = tf.zeros((batch_size, int(self.args["Embedding"]["word_embedding_size"]) ), dtype=tf.float32)   
            combined_attns_state = tf.zeros((batch_size, 4*int(self.args["Encoder"]["hidden_size"])),
                                   dtype=tf.float32)
        

        initial_states["combined_attns_state"] = combined_attns_state
        initial_states["prev_decoder_attn_state"] = prev_decoder_attn_state
        initial_states["attns_state_different"] = attns_state_different
        self.initial_states = initial_states
        return initial_states


    def _init_decoder_params_beamsearch(self, sess, encoder_state, encoder_state_val , distract_shape=None):
        initial_states = {}
        distract_shape = encoder_state.get_shape()[-1].value
        print ("Distract State: ",distract_shape)
        feed_dict = {encoder_state: encoder_state_val}
        with tf.variable_scope("Decoder_Copy", reuse = tf.AUTO_REUSE):
            with tf.variable_scope("Init_Decoder_State", reuse=tf.AUTO_REUSE):
                encoder_state_reshaped = self.fully_connected(encoder_state, int(self.args["Decoder"]["hidden_size"]), activation_fn=None, keep_prob=self.dropout_param)
                #decoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state_reshaped, encoder_state_reshaped)
                initial_states["decoder_state_h"] = encoder_state_reshaped
                initial_states["decoder_state_c"] = encoder_state_reshaped


            attns_state = tf.zeros(shape=(int(self.args["BeamSearch"]["batch_size"]), distract_shape))
            initial_states["attns_state"] = attns_state

        to_return = sess.run(initial_states, feed_dict= feed_dict)  
        return initial_states, to_return

    def compute_gated_context_vector(new_attns_state, attention_on_decoder_states, memory_states, prev_decoder_states):

        memory_state = tf.reduce_sum(memory_states, axis=1)
        prev_decoder_state = tf.reduce_sum(prev_decoder_states, axis=1)
        with tf.variable_scope("gated_projection", reuse=tf.AUTO_REUSE):
            temp_output = self.fully_connected(tf.concat([memory_state, prev_decoder_state], axis=-1), int(self.args["Decoder"]["hidden_size"]), activation_fn=tf.nn.tanh)
        with tf.variable_scope("gate_context_vector", reuse=tf.AUTO_REUSE):
            gated_memory_prev_decoder = self.fully_connected(temp_output, 1, activation_fn=tf.nn.sigmoid)
        
        return new_attns_state * ( 1 - gated_memory_prev_decoder) + gated_memory_prev_decoder * attention_on_decoder_states
        

    def decode_one_step_first_decoder(self, encoder_input_batch, memory_states, prev_time_step_data, query_state,word_embeddings,decoder_cell, content_weights,
                        content_prop_weights, word_indices, prop_indices, is_inference=False, var_name="", is_prev_decoder=True):
        # This function given previous state, and list of current tokens should be able to predict the next set of tokens and the new states

        with tf.variable_scope("Decoder_Copy{}".format(var_name), reuse = tf.AUTO_REUSE):
          prev_coverage = self.args["Decoder"]["use_coverage"] == "True"

          prev_tokens = prev_time_step_data["token"]
          prev_state_temp   = tf.contrib.rnn.LSTMStateTuple(prev_time_step_data["state_temp_c"], prev_time_step_data["state_temp_h"])
          prev_state   = tf.contrib.rnn.LSTMStateTuple(prev_time_step_data["state_c"], prev_time_step_data["state_h"])
          
          prev_coverage_vec = prev_time_step_data["prev_coverage"]#tf.zeros((batch_size, memory_states.get_shape()[1].value))

          print ("Prev Coverage Vec size ", prev_coverage_vec.get_shape())
          prev_attns_state = prev_time_step_data["attns_state"]

          prev_token_embeddings = tf.reshape(tf.nn.embedding_lookup(word_embeddings, prev_tokens),
                                                                    shape=(-1, int(self.args["Embedding"]["word_embedding_size"])))
          
          print ("In Decode One Step",query_state.get_shape())
  		    
          input_size = int(self.args["Embedding"]["word_embedding_size"])
          temp_x = [prev_token_embeddings, prev_attns_state]

          with tf.variable_scope("DecoderInput", reuse=tf.AUTO_REUSE):
              print ("Attention shape:", prev_attns_state.get_shape(), "Query State: ",query_state.get_shape(), decoder_cell[0], decoder_cell[1])

              if (self.args["Query"]["use_query"] == "True"):
                  temp_x.append(query_state)

              if (self.args["Decoder"]["use_concat"] == "True"):	
                  x = tf.concat(temp_x, axis=-1)
              else:
                  x =  self.fully_connected(tf.concat(temp_x, axis=-1), input_size, activation_fn=None, keep_prob=self.dropout_param)
              cell_output_temp, new_state_temp = decoder_cell[0](x, prev_state_temp)

          if (self.args["Decoder"].get("use_single_layer") == "True"):
              cell_output, new_state = cell_output_temp, new_state_temp

          else:  
            with tf.variable_scope("DecoderInput1", reuse=tf.AUTO_REUSE):
                cell_output, new_state = decoder_cell[1](cell_output_temp, prev_state)

          query_vec = [cell_output] 
          if (self.args["Query"]["use_query"] == "True"):
              query_vec.append(query_state)

          if prev_coverage:
             prev_coverage_vec_temp = prev_coverage_vec
          else:
             prev_coverage_vec_temp = None

          new_attns_state, attention_values, _ = self.compute_multi_head_attention("decoder", memory_states, self.masked_weights,
                                                                                tf.concat(query_vec, axis=-1), return_all=False, num_heads=1, prev_coverage_vec=prev_coverage_vec_temp)
                                                                                #property_attention_weights = property_attn_weights)


          # Add gating between passage new_attns_state and question encoder.
          #new_attns_state = self.compute_gated_context_vector(new_attns_state, attention_on_decoder_states, memory_states, prev_decoder_states)

          if prev_coverage:
             prev_coverage_vec  = prev_coverage_vec + attention_values[0]
          else:
             prev_coverage_vec = None


          with tf.variable_scope("AttnOutputProjection", reuse=tf.AUTO_REUSE):

              # output projection consists of cell_output, context vector
              temp_o = [cell_output, new_attns_state[0]] 
              if (self.args["Decoder"]["use_word_embed_outputproj"] == "True"):	
                  temp_o.append(prev_token_embeddings)

              output =  self.fully_connected(tf.concat(temp_o, axis=-1), 
                                                         int(self.args["Decoder"]["hidden_size"]), activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
              
              vocab_projection = tf.nn.softmax(tf.nn.xw_plus_b(output, self.output_projection[0], self.output_projection[1]), axis=1)

              # gate input :
              temp_g = [cell_output, prev_token_embeddings, new_attns_state[0]]
              if (self.args["Query"]["use_query"] == "True"):
                  temp_g.append(query_state)

              output_switch_full = self.generator_switch(tf.concat(temp_g, axis=-1))
              
              print ("DEBUG: vocab_projection {}".format(vocab_projection))
              
              comb_projection, _ = self._get_the_comb_projection(vocab_projection, attention_values, output_switch_full, content_weights, word_indices = word_indices,is_inference=is_inference)
 
        self.return_values_one_step =  { "state": new_state, 
                 "state_temp": new_state_temp, 
                "attns_state":new_attns_state[0],
                "attention_values": attention_values, 
                "comb_projection":comb_projection,
                "output_switch" : output_switch_full,
                "prev_coverage":prev_coverage_vec,
                "attention_values_prop": tf.zeros([1]),
                "prev_coverage_between_decoders":tf.zeros([1]),
                "prev_coverage_vec_different":tf.zeros([1]),
                "prev_decoder_attn_state": tf.zeros([1]), 
                "attns_state_different": tf.zeros([1]),
                "attention_on_decoder_states": tf.zeros([1]),
                "combined_attns_state":tf.zeros([1]), 
                 "attention_values_different": tf.zeros([1]), }


        return self.return_values_one_step



    def decode_one_step_second_decoder(self, encoder_input_batch, memory_states, prev_time_step_data, query_state,word_embeddings,decoder_cell, content_weights,
                        content_prop_weights, word_indices, prop_indices, is_inference=False, var_name="", is_prev_decoder=True,
                         masked_weights_props = None, prev_decoder_mean_attn=None, question_gate=None):
        # This function given previous state, and list of current tokens should be able to predict the next set of tokens and the new states

        with tf.variable_scope("Decoder_Copy{}".format(var_name), reuse = tf.AUTO_REUSE):
          print("IIIIIIIIIIIIIIIII", prev_decoder_mean_attn)
          prev_coverage = self.args["Decoder"]["use_coverage"] == "True"

          prev_tokens = prev_time_step_data["token"]
          prev_state_temp   = tf.contrib.rnn.LSTMStateTuple(prev_time_step_data["state_temp_c"], prev_time_step_data["state_temp_h"])
          prev_state   = tf.contrib.rnn.LSTMStateTuple(prev_time_step_data["state_c"], prev_time_step_data["state_h"])
          
          prev_coverage_vec = prev_time_step_data["prev_coverage"]
          prev_coverage_between_decoders = prev_time_step_data["prev_coverage_between_decoders"]

          prev_coverage_vec_different = prev_time_step_data["prev_coverage_vec_different"]
          print ("Prev Coverage Vec size ", prev_coverage_vec.get_shape())
          
          # combine the decoder1 and encoder attention states for Decoder2. In presence of gate, it will be weighted
          # else it will be a concatenation
          combined_prev_attns_state = prev_time_step_data["combined_attns_state"]
          
          attns_state_different = prev_time_step_data["attns_state_different"]

          prev_decoder_states = prev_time_step_data["prev_decoder_states"]

          #TO DO : Add character embeddings for the generated word 

          prev_token_embeddings = tf.reshape(tf.nn.embedding_lookup(word_embeddings, prev_tokens),
                                                                    shape=(-1, int(self.args["Embedding"]["word_embedding_size"])))
          
          print ("In Decode One Step",query_state.get_shape())
  		    
          input_size = int(self.args["Embedding"]["word_embedding_size"])
          seq_length = memory_states.get_shape()[-1].value
          
          #temp_x denotes input to the decoder
          # Input to the decoder is the : answer, previous predicted word (embedding), combined context vector, and context vector different from the passage
          # context vector.
          if (self.args["Query"]["use_query"] == "True"):
              temp_x = [query_state]
          else:
              temp_x = []

          temp_x += [prev_token_embeddings, combined_prev_attns_state]

          if self.args["Decoder"]["use_attention_different"] == "True":
              temp_x.append(attns_state_different)


          with tf.variable_scope("DecoderInput", reuse=tf.AUTO_REUSE):
              print ("Attention shape:", combined_prev_attns_state.get_shape(), "Query State: ",query_state.get_shape(), decoder_cell[0], decoder_cell[1])
             
              if (self.args["Decoder"]["use_concat"] == "True"):	
                  x = tf.concat(temp_x, axis=-1)
              else:
                  x =  self.fully_connected(tf.concat(temp_x, axis=-1), input_size, activation_fn=None, keep_prob=self.dropout_param)
              cell_output_temp, new_state_temp = decoder_cell[0](x, prev_state_temp)

          # Stacked Decoder
          if (self.args["Decoder"].get("use_single_layer") == "True"):
              cell_output, new_state = cell_output_temp, new_state_temp
          else:  
            with tf.variable_scope("DecoderInput1", reuse=tf.AUTO_REUSE):
                cell_output, new_state = decoder_cell[1](cell_output_temp, prev_state)

          query_vec = [cell_output] 

          if (self.args["Query"]["use_query"] == "True"):
              query_vec.append(query_state)
            
          attention_on_decoder_values = []

          with tf.variable_scope("Attention_Prev_Decoder", reuse=tf.AUTO_REUSE):
            #prev_decoder_states = tf.stack(prev_decoder_states, axis=1)
            print ("PREV DECODER STATES", prev_decoder_states)
            attention_on_decoder_states, attention_on_decoder_values, _  = self.compute_multi_head_attention("prev_decoder_attention", prev_decoder_states, 
                                                                        masked_weights_props, tf.concat(query_vec, axis=-1), return_all=False, num_heads=1,
                                                                        prev_coverage_vec=prev_coverage_between_decoders)
        
            # memnet changes should not be done for attention on decoder values
            attention_on_decoder_states = attention_on_decoder_states[0]
            attention_on_decoder_values = attention_on_decoder_values[0]

          
          if prev_coverage:
             prev_coverage_vec_temp = prev_coverage_vec
          else:
             prev_coverage_vec_temp = None

          new_attns_state, attention_values, _ = self.compute_multi_head_attention("decoder", memory_states, self.masked_weights,
                                                                                tf.concat(query_vec, axis=-1), return_all=False, num_heads=1, prev_coverage_vec=prev_coverage_vec_temp)
                                                                                #property_attention_weights = property_attn_weights)


          # Add gating between passage new_attns_state and question encoder.

          # projection both the context vector to the same space/
          with tf.variable_scope("decoder_states_proj", reuse=tf.AUTO_REUSE):
                attention_on_decoder_states =  self.fully_connected(attention_on_decoder_states, 
                                                    2*int(self.args["Decoder"]["hidden_size"]), activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
          with tf.variable_scope("new_attns_state_proj", reuse=tf.AUTO_REUSE):
                new_attns_state = [self.fully_connected(new_attns_state[0], 2*int(self.args["Decoder"]["hidden_size"]), activation_fn = tf.nn.tanh,
                             keep_prob = self.dropout_param)]

          if self.args["Question_encoder"]["use_word_level_gate"] == "True":
              question_gate = self.gate_soft(tf.concat([attention_on_decoder_states, new_attns_state[0], cell_output], axis=-1))
              question_gate = tf.expand_dims(question_gate, axis=1)             
              combined_new_attns_state = [(question_gate)* attention_on_decoder_states   + (1 - question_gate)*new_attns_state[0]]

          elif self.args["Question_encoder"]["use_gate_policy"] == "True": #self.args["Question_encoder"]["use_question_encoder"] == "True"
              question_gate = tf.expand_dims(question_gate, axis=1)
              combined_new_attns_state = [(question_gate)* attention_on_decoder_states   + (1 - question_gate)*new_attns_state[0]]
          else:
              combined_new_attns_state = [tf.concat([new_attns_state[0], attention_on_decoder_states], axis=-1)]

          # Compute Context Vector different from passage that is already attended using decoder1
          print("IIIIIIIIIIIIIIIII", prev_decoder_mean_attn)
          if (self.args["Decoder"]["use_attn_product"] != "False"):
              prev_decoder_mean_attn = None
  

          print("IIIIIIIIIIIIIIIII", prev_decoder_mean_attn)
          attns_state_different, attention_values_different, _ = self.compute_multi_head_attention("decoder_diff", memory_states, self.masked_weights, 
                                                                                                    tf.concat(query_vec, axis=-1), return_all=False, num_heads=1, 
                                                                                                    prev_coverage_vec = prev_coverage_vec_different,prev_decoder_mean_attn= prev_decoder_mean_attn)

          #Attend on something different than where decoder 1 attended.
          
          prev_coverage_vec_different += attention_values_different[0]

          if prev_coverage:
             prev_coverage_vec  = prev_coverage_vec + attention_values[0]
          else:
             prev_coverage_vec = None

          prev_coverage_between_decoders = prev_coverage_between_decoders + attention_on_decoder_values


          with tf.variable_scope("AttnOutputProjection", reuse=tf.AUTO_REUSE):

              # For output projection, we use  the decoder output, and the context vectors.
              temp_o = [cell_output, combined_new_attns_state[0]]

              if (self.args["Decoder"]["use_attention_different"] == "True"):	
                  temp_o.append(attns_state_different[0])

              if (self.args["Decoder"]["use_word_embed_outputproj"] == "True"):	
                  temp_o.append(prev_token_embeddings)

              if (self.args["Query"]["use_query"] == "True"):
                  temp_o.append(query_state)

              print (temp_o)
              output =  self.fully_connected(tf.concat(temp_o, axis=-1), 
                                                         int(self.args["Decoder"]["hidden_size"]), activation_fn=tf.nn.tanh, keep_prob=self.dropout_param)
              

              # To use different or same output projection layer
              if self.args["Decoder"]["use_diff_out_proj"] == "True":
                  vocab_projection = tf.nn.softmax(tf.nn.xw_plus_b(output, self.prev_output_projection[0], self.prev_output_projection[1]), axis=1)
              else:
                  vocab_projection = tf.nn.softmax(tf.nn.xw_plus_b(output, self.output_projection[0], self.output_projection[1]), axis=1)


              # To toggle between copy and generation
              temp_g = [cell_output, prev_token_embeddings, combined_new_attns_state[0]]

              if (self.args["Decoder"]["use_attention_different"] == "True"):	
                  temp_g.append(attns_state_different[0])

              if (self.args["Query"]["use_query"] == "True"):
                  temp_g.append(query_state)
              # Use a three level gate to switch between three distributions

             
              output_switch_full = self.generator_switch_prop_3gates(tf.concat(temp_g, axis=-1))
              
              print ("DEBUG: vocab_projection {}".format(vocab_projection))
              seq_length = memory_states.get_shape()[1].value

              if (self.args["Decoder"]["use_attention_different"] == "True"):
                  comb_projection, total_attention = self._get_the_comb_projection_property_3gates(vocab_projection, attention_values, attention_values_different, 
                                                                              output_switch_full, content_weights,word_indices=word_indices,is_inference=is_inference)

              else:
                  output_switch_full = self.generator_switch(tf.concat(temp_g, axis=-1))
                  comb_projection, total_attention = self._get_the_comb_projection(vocab_projection, attention_values, output_switch_full, content_weights, word_indices = word_indices,is_inference=is_inference)

        self.return_values_one_step =  { "state": new_state, 
                 "state_temp": new_state_temp, 
                "combined_attns_state":combined_new_attns_state[0],
                "attention_values": attention_values, 
                "comb_projection":comb_projection,
                "output_switch" : output_switch_full,
                "prev_coverage":prev_coverage_vec,
                "attention_values_prop": attention_on_decoder_values,
                "prev_coverage_between_decoders":prev_coverage_between_decoders,
                "prev_coverage_vec_different":prev_coverage_vec_different, 
                "attns_state_different": attns_state_different[0],
                "attention_values_different": attention_values_different[0], 
                "total_attention": [total_attention]}

        return self.return_values_one_step

    def decoder_with_copy(self,encoder_input_batch, decoder_input_batch, loop_function=None, is_prev_decoder=True, prev_decoder_states=None, prop_indices = None,
                         prev_coverage_vec_different=None, masked_weights_props = None, prev_decoder_mean_attn=None, question_gate=None):
        
        self.decoder_cell = []
        var_name = ""
        if is_prev_decoder:
            with tf.variable_scope("Decoder_Copy_Cell", reuse=tf.AUTO_REUSE):
                self.decoder_cell.append(self.select_cell( "decoder_cell", self.args["Decoder"]["cell_type"], int(self.args["Decoder"]["hidden_size"]), 1,  dropout_param=self.dropout_param)[0])

            with tf.variable_scope("Decoder_Copy_Cell_Second", reuse=tf.AUTO_REUSE):
                self.decoder_cell.append(self.select_cell( "decoder_cell1", self.args["Decoder"]["cell_type"], int(self.args["Decoder"]["hidden_size"]), 1, dropout_param=self.dropout_param)[0])
        else:
            var_name = "new_decoder"
            #var_name=""
            with tf.variable_scope("Decoder_Copy_Cell_New", reuse=tf.AUTO_REUSE):
                self.decoder_cell.append(self.select_cell( "decoder_cell", self.args["Decoder"]["cell_type"], int(self.args["Decoder"]["hidden_size"]), 1,  dropout_param=self.dropout_param)[0])

            with tf.variable_scope("Decoder_Copy_Cell_Second_New", reuse=tf.AUTO_REUSE):
                self.decoder_cell.append(self.select_cell( "decoder_cell1", self.args["Decoder"]["cell_type"], int(self.args["Decoder"]["hidden_size"]), 1, dropout_param=self.dropout_param)[0])
       
        if is_prev_decoder:
           print ("--"*30, "Preksha: In Second Decoder")
           is_prev_state = None
           prev_coverage_vec_different = tf.zeros([1])

        else:
           prev_decoder_state = tf.reduce_mean(prev_decoder_states, axis=1)
           is_prev_state = tf.concat([self.encoder_state, prev_decoder_state], axis=-1)


        initial_states = self._init_decoder_params(self.memory_states.get_shape()[-1].value, is_prev_state=is_prev_state, var_name=var_name)

        batch_size = int(self.args["Hyperparams"]["batch_size"])
        max_length = int(self.args["Encoder"]["max_sequence_length"])
        
        state =   tf.contrib.rnn.LSTMStateTuple(initial_states["decoder_state_c"], initial_states["decoder_state_h"])
        state_temp = tf.contrib.rnn.LSTMStateTuple(initial_states["decoder_state_c"], initial_states["decoder_state_h"])
        
        attns_state = initial_states["attns_state"] 
        combined_attns_state = initial_states["combined_attns_state"]
        prev_decoder_attn_state = initial_states["prev_decoder_attn_state"]
        attns_state_different = initial_states["attns_state_different"]

        prev_coverage = tf.zeros((batch_size, max_length), dtype=tf.float32)
        prev_coverage_between_decoders = tf.zeros((batch_size, int(self.args["Decoder"]["max_prop_steps"])-1), tf.float32)

        token_predicted=None
        attention_weights = []
        attention_weights_prop = []
        outputs = []
        predicted_tokens = []
        output_switches = []
        attention_values_different = []
        self.prev_time_step_data ={} 
        beta_weights = []
        orig_outputs = []

        decoder_inputs = tf.unstack(decoder_input_batch["word"], axis=1)
        if is_prev_decoder:
           decoder_inputs = tf.unstack(decoder_input_batch["property_word"], axis=1)

        for i, inp in enumerate(decoder_inputs):
            
            print ("-"*30 + "\n", i)
            print (prev_coverage.get_shape())
            if loop_function is not None and token_predicted is not None:
                inp = token_predicted
           
            self.prev_time_step_data["token"] = inp
            self.prev_time_step_data["attns_state"] = attns_state
            self.prev_time_step_data["combined_attns_state"] = combined_attns_state
            self.prev_time_step_data["state_c"] = state.c
            self.prev_time_step_data["state_temp_c"] = state_temp.c
            self.prev_time_step_data["state_h"] = state.h
            self.prev_time_step_data["state_temp_h"] = state_temp.h
            self.prev_time_step_data["prev_coverage"] = prev_coverage
            self.prev_time_step_data["prev_decoder_attn_state"] = prev_decoder_attn_state
            self.prev_time_step_data["attns_state_different"] = attns_state_different
            self.prev_time_step_data["prev_coverage_between_decoders"] = prev_coverage_between_decoders
            self.prev_time_step_data["prev_coverage_vec_different"] = prev_coverage_vec_different

            var_name = ""
            if not is_prev_decoder:
               self.prev_time_step_data["prev_decoder_states"] = prev_decoder_states
               var_name = "new_decoder"
 
            if is_prev_decoder:
                new_state_values = self.decode_one_step_first_decoder(encoder_input_batch,self.memory_states, self.prev_time_step_data,
                               self.query_state, self.word_embeddings, self.decoder_cell,content_weights = encoder_input_batch["content_weights"],
                               content_prop_weights = decoder_input_batch["content_prop_weights"], 
                               word_indices = encoder_input_batch["word"], prop_indices=prop_indices, var_name=var_name, is_prev_decoder=is_prev_decoder)
            else:
                new_state_values = self.decode_one_step_second_decoder(encoder_input_batch,self.memory_states, self.prev_time_step_data,
                               self.query_state, self.word_embeddings, self.decoder_cell,content_weights = encoder_input_batch["content_weights"],
                               content_prop_weights = decoder_input_batch["content_prop_weights"], 
                               word_indices = encoder_input_batch["word"], prop_indices=prop_indices, var_name=var_name, is_prev_decoder=is_prev_decoder,
                                masked_weights_props = masked_weights_props, prev_decoder_mean_attn=prev_decoder_mean_attn, question_gate=question_gate)                    

            # print ("-"*30 ,i,inp_pos.get_shape())

            state = new_state_values["state"]
            state_temp = new_state_values["state_temp"]
            attns_state = new_state_values.get("attns_state")
            combined_attns_state = new_state_values["combined_attns_state"]
            comb_projection = new_state_values["comb_projection"]
            output_switch = new_state_values["output_switch"]
            attention_values = new_state_values["attention_values"]
            attention_values_prop = new_state_values["attention_values_prop"]
            beta_values = new_state_values.get("property_attn_values")
            prev_coverage = new_state_values["prev_coverage"]
            property_state = new_state_values.get("property_state")
            prev_decoder_attn_state = new_state_values.get("prev_decoder_attn_state")
            prev_coverage_between_decoders = new_state_values["prev_coverage_between_decoders"]
            attns_state_different = new_state_values["attns_state_different"] 

            if not is_prev_decoder:
                prev_coverage_vec_different = new_state_values.get("prev_coverage_vec_different")
                attns_state_different = new_state_values.get("attns_state_different")

            print (new_state_values)
            print ("Attention_state different", attns_state_different)
            outputs.append(comb_projection)
            orig_outputs.append(state.h)
            attention_values = tf.stack(attention_values, axis=0)
            attention_values = tf.reduce_mean(attention_values, axis=0)
            attention_values_different.append(new_state_values.get("attention_values_different"))
            attention_weights.append(attention_values)
            attention_weights_prop.append(attention_values_prop)
            output_switches.append(output_switch)
            beta_weights.append(beta_values)

            if self.is_sample:
                if "number_of_samples" in self.args["Hyperparams"] and int(self.args["Hyperparams"]["number_of_samples"]) >= 1:
                    token_predicted = tf.argmax(tf.distributions.Multinomial(float(self.args["Hyperparams"]["number_of_samples"]),
                                                probs=comb_projection).sample(1), axis=-1)
                    
                token_predicted = tf.distributions.Categorical(probs=comb_projection).sample()
                #token_predicted = tf.random.multinomial(comb_projection, 1)
            else:
                token_predicted = tf.argmax(comb_projection, axis=1)

            predicted_tokens.append(token_predicted)

        if self.args["Encoder"]["use_property_attention"] == "False":
            beta_weights = attention_weights
        new_state_values["attention_values_prop"] = attention_weights_prop
        new_state_values["output_switch"] = output_switches
        new_state_values["predicted_tokens"] = predicted_tokens
        new_state_values["attention_values_different"] = attention_values_different
        return outputs, state, attention_weights, orig_outputs, predicted_tokens, beta_weights, initial_states, new_state_values



    def decoder_wrapper(self, encoder_input_batch, decoder_input_batch, encoder_state, memory_states, masked_weights, query_state,
        feed_previous, is_prev_decoder=True, prev_decoder_state=None, prop_indices=None, prev_coverage_vec_different=None,
        masked_weights_props = None, prev_decoder_mean_attn=None, question_gate=None):
        print ("In decoder wrapper")
        if self.args["Decoder"]["decoder_type"]== "vad":
            loop_function = self._extract_argmax_and_embed(False) if feed_previous else None
            print ("loop_function", loop_function)
            return self.decoder_with_copy( encoder_input_batch, decoder_input_batch, loop_function=loop_function, is_prev_decoder = is_prev_decoder,
                                        prev_decoder_states=prev_decoder_state, prop_indices=prop_indices, prev_coverage_vec_different=prev_coverage_vec_different,
                                        prev_decoder_mean_attn=prev_decoder_mean_attn, question_gate=question_gate, masked_weights_props=masked_weights_props)

        elif self.args["Decoder"]["decoder_type"] == "copy":
            loop_function = self._extract_argmax_and_embed(False) if feed_previous else None
            print ("loop_function", loop_function)
            return self.decoder_with_copy( encoder_input_batch, decoder_input_batch, loop_function = loop_function, is_prev_decoder=is_prev_decoder,
                                           prev_decoder_states=prev_decoder_state, prop_indices=prop_indices, prev_coverage_vec_different=prev_coverage_vec_different,
                                           prev_decoder_mean_attn=prev_decoder_mean_attn, question_gate=question_gate, masked_weights_props=masked_weights_props)
        else:
            loop_function = None
            print ("loop_function", loop_function)
            return self.decoder_with_copy(encoder_input_batch, decoder_input_batch,  loop_function=loop_function,
                                          prev_decoder_states=prev_decoder_state, prop_indices=prop_indices, prev_coverage_vec_different=prev_coverage_vec_different,
                                          prev_decoder_mean_attn=prev_decoder_mean_attn, question_gate=question_gate, masked_weights_props=masked_weights_props)


    def decoder_per_feed_previous(self,  encoder_input_batch, decoder_input_batch, encoder_state, memory_states, masked_weights, query_state,
                                  feed_previous=None, is_prev_decoder=True, prev_decoder_state=None, prop_indices=None, prev_coverage_vec_different=None,
                                 masked_weights_props = None, prev_decoder_mean_attn=None, question_gate=None):
        print ("feed previous",feed_previous)                          
        if isinstance(feed_previous, bool):
            return self.decoder_wrapper(encoder_input_batch, decoder_input_batch, encoder_state, memory_states, masked_weights, query_state, feed_previous, is_prev_decoder,
                                        prev_decoder_state=prev_decoder_state, prev_coverage_vec_different=prev_coverage_vec_different,
                                        prev_decoder_mean_attn=prev_decoder_mean_attn, question_gate=question_gate, masked_weights_props=masked_weights_props)

        def temp_decoder(feed_previous_bool):
            outputs, state, attention_weights, output_switches, predicted_tokens, beta_weights, initial_states, new_state_values  = self.decoder_wrapper(encoder_input_batch,
                                                                                                        decoder_input_batch, 
                                                                                                        encoder_state, 
                                                                                                        memory_states, 
                                                                                                        masked_weights, 
                                                                                                        query_state, feed_previous_bool, is_prev_decoder,
                                                                                                        prev_decoder_state=prev_decoder_state, prop_indices=prop_indices,
                                                                                                        prev_coverage_vec_different=prev_coverage_vec_different,
                                                                                                        masked_weights_props = masked_weights_props,
                                                                                                        prev_decoder_mean_attn=prev_decoder_mean_attn, question_gate=question_gate)

            state_list = [state]

            return outputs + state_list + attention_weights + output_switches + predicted_tokens + beta_weights + [ [initial_states, new_state_values] ]

        outputs_state_aw = tf.cond(feed_previous,
                                   lambda: temp_decoder(True), 
                                   lambda: temp_decoder(False))
        
        return outputs_state_aw
