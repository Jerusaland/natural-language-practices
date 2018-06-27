import os
import time
import numpy as np
import tensorflow as tf

def bi_lstm_model(input_tensor,batch_size,words_quantity):
    
    hidden_units = 256
    dropout_rate = 0.1
    
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(hidden_units)
    lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, input_keep_prob=(1-dropout_rate))
        
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hidden_units)
    lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(lstm_cell_2, input_keep_prob=(1-dropout_rate))
        
    stacked_lstm = [lstm_cell_1]
    stacked_lstm.append(lstm_cell_2)
    m_lstm_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm, state_is_tuple = True)
        
    (outputs, state) = tf.nn.dynamic_rnn(m_lstm_cell, input_tensor, dtype=tf.float32)

    outputs = tf.reshape(outputs, [-1, hidden_units])
    #weights & bias
    W = tf.Variable(tf.truncated_normal([hidden_units,words_quantity],stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[words_quantity]))
        
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_size,-1,words_quantity])
    return logits


#define the bi-GRU model
def bi_gru_model(input_tensor,batch_size,words_quantity):
    
    hidden_units = 256
    dropout_rate = 0.1

    cell_1 = tf.contrib.rnn.GRUCell(hidden_units)
    cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, input_keep_prob=(1-dropout_rate))
        
    cell_2 = tf.contrib.rnn.GRUCell(hidden_units)
    cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2, input_keep_prob=(1-dropout_rate))
        
    stacked_gru = [cell_1]
    stacked_gru.append(cell_2)
    m_gru_cell = tf.contrib.rnn.MultiRNNCell(stacked_gru, state_is_tuple = True)
        
    (outputs, state) = tf.nn.dynamic_rnn(m_gru_cell, input_tensor, dtype=tf.float32)
    
    outputs = tf.reshape(outputs, [-1, hidden_units])
    #weights & bias
    W = tf.Variable(tf.truncated_normal([hidden_units,words_quantity],stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[words_quantity]))
        
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_size,-1,words_quantity])
    return logits   

#define a 3 layers GRU model
def tri_gru_model(input_tensor,batch_size,words_quantity):
    
    hidden_units = 256
    dropout_rate = 0.1

    def gru_cell():
        layer = tf.contrib.rnn.GRUCell(hidden_units)
        layer = tf.contrib.rnn.DropoutWrapper(layer, input_keep_prob=(1-dropout_rate))
        return layer
        
    m_gru_cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for i in range(3)], state_is_tuple=True)
    (outputs, state) = tf.nn.dynamic_rnn(m_gru_cell, input_tensor, dtype=tf.float32)
        
    outputs = tf.reshape(outputs, [-1, hidden_units])
    #weights & bias
    W = tf.Variable(tf.truncated_normal([hidden_units,words_quantity],stddev=0.1))
    b = tf.Variable(tf.constant(0.0, shape=[words_quantity]))
        
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_size,-1,words_quantity])     
    return logits