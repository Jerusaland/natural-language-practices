import os
import time
import numpy as np
import tensorflow as tf


#define the validation processes
def validate_process():
    
    batch_size = 1
    mfcc_dims = 20

    wav_path = 'd:\practices\STT_Dataset\test'
    text_path = 'd:\practices\STT_Dataset\data'
    
    #get the validation wav file list and corresponding texts
    wav_files,true_texts, _ = get_wav_and_text(wav_path, text_path)
	
	#get words bag
	words,_,_,_,_ = get_text_features(true_texts)
	words_quantity = len(words) # blank space included

    graph = tf.Graph()
    with graph.as_default():    
		#add place holder for X(input sequences) and Y(true text labels)
		X = tf.placeholder(dtype=tf.float32, shape=[batch_size,None,mfcc_dims])
		Y = tf.sparse_placeholder(dtype=tf.int32)
		seq_len = tf.placeholder(dtype=tf.int32)
    
		#get neural network output - using placeholder
		logits = bi_gru_model(X, batch_size, words_quantity)
		logits = tf.transpose(logits, perm=[1, 0, 2])
	
        #CTC decode for computing training error
        decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
 	
    saver = tf.train.Saver()    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        for batch in range(5):
            #get the batch audio sequences and sparse tensor
            batch_seq,batch_seq_len,sparse_tensor = get_next_batches(wav_files,true_texts,batch_size,batch)
			
			feed={X: batch_seq, Y: sparse_tensor, seq_len: batch_seq_len}
			d = session.run(decoded[0], feed_dict=feed)

			dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
            dense_labels = utils.trans_sparse_to_texts(sparse_tensor, words)
        
            for orig, decoded_array in zip(dense_labels, dense_decoded):
                # 转成string
                decoded_str = utils.trans_array_to_text_ch(decoded_array, words)
                print('语音原始文本: {}'.format(orig))
                print('识别出来的文本:  {}'.format(decoded_str))
				print('\n')

