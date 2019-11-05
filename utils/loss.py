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

def loss_op(logits, labels, weights, is_copy=True, batch_size=64):
    print ("is_copy")
    _labels  = tf.unstack(labels, axis=1)
    weights  = tf.to_float(weights)
    losses_per_step =[]
    _weights = tf.unstack(weights, axis=1)
    batch_nums = tf.reshape(tf.range(0, limit=batch_size), shape=(-1,1))
    print (batch_nums.get_shape(), _labels[0].get_shape())
    if is_copy:
       for i, step in enumerate(logits):
          label = tf.reshape(tf.stack((batch_nums, tf.reshape(_labels[i], shape=(-1,1))), axis=2), shape=(batch_size, 2))
          print ("label_shape is ", label.get_shape(), step)
          gold_probs = tf.gather_nd(step, label)
          losses = -1* tf.log(gold_probs + 1e-5)
          losses_per_step.append(losses * _weights[i])
       print (losses_per_step[0].get_shape())
       losses_per_step = tf.stack(losses_per_step,axis=1)
       losses_per_step = tf.reduce_sum(losses_per_step,axis=1)/tf.reduce_sum(weights,axis=1)
       losses_per_step = tf.reduce_mean(losses_per_step) #/ batch_size
       tf.summary.scalar('Cross Entropy Loss',losses_per_step)
       return losses_per_step
    else:
       loss_per_batch = tf.contrib.seq2seq.sequence_loss(tf.stack(logits, axis=1), labels, weights, softmax_loss_function=None )

    return loss_per_batch

def loss_question_label(logits, question_labels, question_label_positions, batch_size=32):
    logits = tf.stack(logits, axis=1)
    batch_nums = tf.range(0, limit=batch_size)
    print ("batch_nums, question_label_positions, question_labels",batch_nums.get_shape(), question_label_positions.get_shape(), question_labels.get_shape())
    indices = tf.stack([batch_nums, question_label_positions, question_labels], axis=1)
    gold_probs = tf.gather_nd(logits, indices)
    weights = tf.cast(tf.greater(tf.cast(question_labels, tf.float32), tf.zeros(batch_size, dtype=tf.float32)), tf.float32)
    losses = -1 * tf.reduce_mean(tf.log(gold_probs + 1e-5)*weights)
    tf.summary.scalar('Question Loss', losses)
    return losses

def entropy(logits, weights):
    weights = tf.to_float(weights)
    losses_per_step = []
    _weights = tf.unstack(weights, axis=1)
    for i, step in enumerate(logits):
        losses = -1 * step * tf.log(step + 1e-5)
        losses_per_step.append(tf.reduce_sum(losses, axis=1) * _weights[i])  # losses will be 30000 in size, needs to be reduced by sum 
    losses_per_step = tf.stack(losses_per_step, axis=1 )
    losses_per_step = tf.reduce_sum(losses_per_step, axis=1)/tf.reduce_sum(weights, axis=1)
    losses_per_step = tf.reduce_mean(losses_per_step) #/batch_size
    return losses_per_step

def sum_logits(logits, weights):
    weights = tf.to_float(weights)
    losses_per_step = []
    _weights = tf.unstack(weights, axis=1)
    for i, step in enumerate(logits):
        losses = step
        losses_per_step.append(tf.reduce_sum(losses, axis=1) * _weights[i])  # losses will be 30000 in size, needs to be reduced by sum 
    losses_per_step = tf.stack(losses_per_step, axis=1 )
    losses_per_step = tf.reduce_sum(losses_per_step, axis=1)/tf.reduce_sum(weights, axis=1)
    losses_per_step = tf.reduce_mean(losses_per_step) #/batch_size
    tf.summary.scalar("Addition of probs", losses_per_step)
    return losses_per_step
    
def information_gain_between_decoders(logits_decoder_2, weights_2, logits_decoder_1, weights_1):
    entropy_decoder_1 = entropy(logits_decoder_1,  weights_1)
    entropy_decoder_2 = entropy(logits_decoder_2,  weights_2)

    tf.summary.scalar("Entropy of 2 given 1", entropy_decoder_2)
    tf.summary.scalar("Entropy of 1", entropy_decoder_1)

    return -1 * (entropy_decoder_1 - entropy_decoder_2) 

def reinforce_loss(args,logits, predicted_tokens, rewards, rewards_baseline, weights, batch_size=64):
    batch_nums = tf.reshape(tf.range(0, limit=batch_size,dtype=tf.int32), shape=(-1,1))
    predicted_tokens =  tf.to_int32(predicted_tokens)
    losses_per_step = []
    weights = tf.to_float(weights) 
    _weights = tf.unstack(weights, axis=1)
    for i, step in enumerate(logits):
        tokens_per_timestep = tf.reshape(tf.stack((batch_nums, tf.reshape(predicted_tokens[i], shape=(-1, 1))), axis=2), shape=(batch_size, 2))
        gold_probs = tf.gather_nd(step, tokens_per_timestep)
        losses =  1 * tf.log(gold_probs + 1e-5)
        losses_per_step.append(losses * _weights[i]) 
    print ("DEBUG: Reward Baseline shape", i,rewards_baseline,"Reward Shape",rewards)
    losses_per_step = tf.stack(losses_per_step, axis=1)
    rewards_baseline = tf.transpose(rewards_baseline)
    rewards = tf.transpose(rewards)
    if ("baseline" in args["Hyperparams"] and args["Hyperparams"]["baseline"] == "False"):
        losses_per_step = (-1*rewards) * losses_per_step
        print ("DEBUG: NO BASELINE")
    else:
        losses_per_step = tf.reduce_mean(rewards_baseline - rewards,axis=1,keep_dims=True) * losses_per_step
    
    losses_per_step  = tf.reduce_mean(losses_per_step)
    tf.summary.scalar('RL Loss',losses_per_step)
    return losses_per_step

def compute_question_gate_rewards(bleu_score,action):
    scale = 1.0
    #threshold = 17.0
    threshold = tf.reduce_mean(bleu_score)
    reward_1 = scale*(bleu_score - threshold)   #action = 1
    reward_0 = scale*(threshold - bleu_score)   # action = 0
    action = tf.cast(action,tf.float32)
    rewards = action*reward_1 + (1-action)*reward_0
    return rewards

def bandits_reinforce(reward, action, policy):
     
     action = tf.cast(action,tf.float32)
     loss = reward* tf.nn.sigmoid_cross_entropy_with_logits(logits=policy,labels=action)
     loss = tf.reduce_mean(loss)
     return loss  

def js_divergence(probs_a,probs_b):

    dist_a = tf.distributions.Categorical(probs=probs_a)
    dist_b = tf.distributions.Categorical(probs=probs_b)
    
    probs_m = (probs_a + probs_b)/2.0
    dist_m = tf.distributions.Categorical(probs=probs_m)

    js = 0.5*(tf.distributions.kl_divergence(dist_a,dist_m) + tf.distributions.kl_divergence(dist_b,dist_m))
    return js


def loss_prop_aw(prop_weights, attn_weights, batch_size=64):
    js_loss = tf.zeros(batch_size)
    #print ("len prop_weights, shape prop_weights, len attn weights, shape of attn_weights",len(prop_weights),prop_weights[0].get_shape(),len(attn_weights),attn_weights[0].get_shape())
    #temp = tf.stack(prop_weights, axis=1)
    #temp = tf.reduce_mean(temp, axis=1)

    for i in range(len(attn_weights)):
        temp = []
        for j in range(len(prop_weights)):
            temp.append(js_divergence(prop_weights[j], attn_weights[i]))
        js_loss += tf.reduce_min(tf.stack(temp, axis=1), axis=1)
        #js_loss += js_divergence(temp, attn_weights[i]) / len(attn_weights)
    final_loss = tf.reduce_mean(js_loss)/len(attn_weights)

    return final_loss

def loss_js(attn_weights, batch_size=64):
    
    js_loss = tf.zeros(batch_size)

    for i in range(len(attn_weights)-1):
        js_loss += js_divergence(attn_weights[i],attn_weights[i+1])
    
    js_loss += js_divergence(attn_weights[0], attn_weights[-1])
    final_loss = tf.reduce_mean(js_loss/ (len(attn_weights)+1)) 
    return final_loss

def loss_coverage(attn_weights, batch_size=64):
    
    coverage_vec = tf.zeros((batch_size, attn_weights[0].get_shape()[-1].value))
    covloss = 0
    for i in attn_weights:
        covloss += tf.reduce_sum(tf.minimum(i, coverage_vec))/(batch_size)
        coverage_vec += i
    covloss = covloss/len(attn_weights) 
    return covloss

def train_op(args,loss, learning_rate, gradient_clip):
    
    if "optimizer" in args["Hyperparams"] and args["Hyperparams"]["optimizer"] == "Momentum": 
        optimizer = tf.train.MomentumOptimizer(learning_rate, momemtum=0.9, use_nesterov=True)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    grads = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1 * gradient_clip, 1* gradient_clip), var) for grad, var in grads]
    train_op = optimizer.apply_gradients(capped_gvs)

    return train_op
