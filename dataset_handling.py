
import os
import numpy as np
from collections import Counter
import librosa
import tensorflow as tf
# import scipy.io.wavfile as wav
# from python_speech_features import mfcc

wav_path = 'd:\practices\STT_Dataset\dev' #音频文件所在目录
text_path = 'd:\practices\STT_Dataset\data' #音频所对应的文字所在目录

# Get audio wave files list and their corresponding texts for training
def get_wav_and_text(wav_dir, text_dir):

    wav_files = []
    text_files = []
#     walk_through = os.walk(wav_dir)
    dirpath, dirnames, filenames = next(os.walk(wav_dir))
    for filename in filenames:
        if filename.endswith('.wav') or filename.endswith('.WAV'):
            wav_file_path = os.path.join(wav_dir, filename)
            wav_files.append(wav_file_path)
        if filename.endswith('.trn') or filename.endswith('.SRN'):
            text_file_path = os.path.join(text_dir, filename)
            text_files.append(text_file_path)
            
    text_dict = {}
    for text_file in text_files:
        file_id = os.path.basename(text_file).split('.')[0]        
        with open(text_file, 'rb') as f:
            line = f.readline()
            text = line.strip(b'\n')
            text_dict[file_id] = text.decode('utf-8')

    new_wav_files = []
    new_texts = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in text_dict:
            wav_id_text = np.array(text_dict[wav_id])
            new_texts.append(wav_id_text)
            new_wav_files.append(wav_file)

    new_wav_files = np.asarray(new_wav_files)
    new_texts = np.asarray(new_texts)
	sample_quantity = len(new_texts)
    
    return new_wav_files, new_texts, sample_quantity


#convert texts to dense tensor, and gather other necessary features
def get_text_features(texts):

    #count all words of text samples, and make a dictionary
    all_words = []
    for text in texts:
        all_words += [word for word in text]
    counter = Counter(all_words)
    words = sorted(counter)
    words_size = len(words)
    word_num_map = dict(zip(words, range(words_size)))
    
    #convert each text sample to dense vector
    texts_tensor = []
    texts_tensor_len = []
    for text in texts:
        target = []
        for word in text:
            word_to_num = word_num_map[word]
            target.append(word_to_num)
#         target = np.asarray(target)
        texts_tensor.append(target)
        texts_tensor_len.append(len(target))
    texts_tensor = np.asarray(texts_tensor)
    texts_tensor_len = np.asarray(texts_tensor_len)
    text_max_len = np.max(texts_tensor_len)

    return words, word_num_map, texts_tensor, texts_tensor_len, text_max_len

#convert text tensor to sparse tensor which is required by CTC
def get_sparse_tensor(texts_tensor, text_max_len=75):
    
    indices = []
    values = []
    for n, seq in enumerate(texts_tensor):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    shape = np.asarray([len(texts_tensor), text_max_len], dtype=np.int64)
    
    return indices,values, shape


#get audio MFCC features using librosa
def get_audio_features_librosa(wav_files):
    audio_features = []
    for file in wav_files:
        wav,sr = librosa.load(file)
        mfcc = np.transpose(librosa.feature.mfcc(wav,sr))
        mfcc = (mfcc - np.mean(mfcc))/np.std(mfcc)
        audio_features.append(mfcc)
    audio_features = np.asarray(audio_features)
    audio_features_len = np.asarray([len(s) for s in audio_features], dtype=np.int64)
    audio_max_len = np.max(audio_features_len)
    mfcc_dims = np.asarray(audio_features[0]).shape[1]
    
    return audio_features,audio_features_len, audio_max_len, mfcc_dims


#音频数据对齐
def pad_sequences(sequences, maxlen=None, mfcc_dims=20, dtype=np.float32,
                  padding='post', truncating='post', value=0.0):
    
    #初始化返回的sequences
    batch_size = len(sequences)
    if maxlen is None:
        maxlen = np.max(np.asarray([len(s) for s in sequences]))
    x = (np.ones((batch_size, maxlen, mfcc_dims)) * value).astype(dtype)
    
    #计算返回的sequences
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


#get batch audio features and corresponding sparse tensor of texts
def get_next_batches(wav_files, text_labels, batch_size=1, n_batch=0):
    
    idx_list = range(n_batch * batch_size, (n_batch+1)* batch_size)
    batch_wav_files = [wav_files[i] for i in idx_list]
    batch_text_labels = [text_labels[i] for i in idx_list]
    
    #获取音频特征，通过get_audio_features函数
    audio_features,audio_features_len, audio_max_len, mfcc_dims = get_audio_features_librosa(batch_wav_files)
    
    #获取文本特征，通过get_text_features函数
    _,_,texts_tensor,texts_tensor_len,text_max_len = get_text_features(batch_text_labels)
       
    #pad audio sequences
    batch_audio_seq = pad_sequences(audio_features,audio_max_len,mfcc_dims)
    
    #获取CTC计算所需要的sparse tensor
    sparse_tensor = get_sparse_tensor(texts_tensor,text_max_len)
    
    return batch_audio_seq, audio_features_len, sparse_tensor 


#把sparse_tensor转换为文字
def trans_sparse_to_texts(tuple, words):
    
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    #print('word len is:' , len(words))
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == 0 else words[c]  # chr(c + FIRST_INDEX)
        results[index] = results[index] + c

    return results


def trans_array_to_text_ch(value, words):
    results = ''
    #print('trans_array_to_text_ch len:', len(value))
    for i in range(len(value)):
        results += words[value[i]]  
    return results.replace('`', ' ')