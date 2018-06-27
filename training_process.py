import os
import time
import numpy as np
import tensorflow as tf


#define the training processes
def train_process():
    
    epochs = 10
    batch_size = 8
    mfcc_dims = 20
    learning_rate = 0.1
    rho=0.9
    epsilon=0.01
    wav_path = 'd:\practices\STT_Dataset\dev'
    text_path = 'd:\practices\STT_Dataset\data'
    
    #get the training wav file list and corresponding texts
    wav_files,true_texts,sample_quantity = get_wav_and_text(wav_path, text_path)
    
    #get the words quantity occured in the training texts samples, and the word-to-number mapping
    words,_,_,_,_ = get_text_features(true_texts)
    words_quantity = len(words) # blank space included
    batch_quantity = int(sample_quantity/batch_size) #how many batches for loop
    
    graph = tf.Graph()
    with graph.as_default():
        #add place holder for X(input sequences) and Y(true text labels)
        X = tf.placeholder(dtype=tf.float32, shape=[batch_size,None,mfcc_dims])
        Y = tf.sparse_placeholder(dtype=tf.int32)
        seq_len = tf.placeholder(dtype=tf.int32)
    
        #get neural network output - using placeholder
        logits = bi_gru_model(X, batch_size, words_quantity)
        #transpose to time_major first
        logits = tf.transpose(logits, perm=[1, 0, 2])
        
        # CTC loss - using placeholder
        loss = tf.nn.ctc_loss(Y, logits, seq_len)
        cost = tf.reduce_mean(loss)
        # optimizer for CTC lost
        optimizer = tf.train.AdadeltaOptimizer(
                     learning_rate=learning_rate,rho=rho,epsilon=epsilon).minimize(cost)
    
        #CTC decode for computing training error
        decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), Y))
        
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
 
        train_start = time.time()
        print("开始训练：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for epoch in range(epochs):
            
            train_cost = 0
            train_err = 0
            
            for batch in range(batch_quantity):
                #get the batch audio sequences and sparse tensor
                batch_seq,batch_seq_len,sparse_tensor = get_next_batches(wav_files,true_texts,batch_size,batch)
                
                feed = {X: batch_seq, Y: sparse_tensor, seq_len: batch_seq_len}
                
                #compute the training cost              
                batch_cost, _ = sess.run([cost, optimizer], feed_dict=feed)
                train_cost += batch_cost
                
                #compute the training error
                batch_err = sess.run(ler, feed_dict=feed)
                train_err += batch_err*batch_size

            train_cost /= sample_quantity
            train_err /= sample_quantity
            
            epoch_duration = time.time() - train_start
            log = '迭代次数 {}/{}, 训练损失: {:.3f}, 错误率: {:.3f}, 时长: {:.2f} sec'
            print(log.format(epoch+1, epochs, train_cost, train_err, epoch_duration))
            
        saver.save(sess, os.getcwd(), global_step=epoch)
        train_duration = time.time() - train_start
    print('Training complete, total duration: {:.2f} min'.format(train_duration / 60))
    
