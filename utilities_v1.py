import numpy
import sklearn
import sys
import os
import random
import re
  
def generate_vocab(dir, min_count, max_files):
    word_dic = {}
    pos_dir = dir + "/pos/"
    neg_dir = dir + "/neg/"
    pos_filenames = [f for f in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, f))]
    
    if max_files == -1:
        num_files = 12500
    else:
        num_files = max_files / 2
         
    i = 0
    while i < int(num_files):
        file = open(pos_dir + pos_filenames[i])
        contents = file.read()
        
        word_list = re.findall(r'\w+', contents)
        
        for word in word_list:
            if not (word == "br"):
                if word in word_dic:
                    word_count = word_dic.get(word) + 1
                else:
                    word_count = 1 
                word_dic.update({word: word_count})       
        i += 1
        
    i = 0
    neg_filenames = [f for f in os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, f))]
    while i < int(num_files):
        file = open(neg_dir + neg_filenames[i])
        contents = file.read()
       
        word_list = re.findall(r'\w+', contents)
        
        for word in word_list:
            if not (word == "br"):
                if word in word_dic:
                    word_count = word_dic.get(word) + 1
                else:
                    word_count = 1 
                word_dic.update({word: word_count})
        i += 1
        
    for key in list(word_dic.keys()):
        value = word_dic[key]
        if value < min_count:
            del word_dic[key]
    
    vocab = list(word_dic.keys())
    return vocab


def create_word_vector(fname, vocab):
    
    feat_vec = numpy.zeros(len(vocab))
    
    contents = fname.read()
    word_list = re.findall(r'\w+', contents)
    
    for word in word_list:
        if not (word == "br"):
            i = 0
            while i < len(vocab):                        
                if vocab[i] == word:
                    feat_vec[i] += 1
                i += 1         
    return feat_vec


def load_data(dir, vocab, max_files):
    pos_dir = dir + "/pos/"
    neg_dir = dir + "/neg/"
    pos_filenames = [f for f in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, f))]
    neg_filenames = [f for f in os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, f))]

    if max_files == -1:
        num_files = 12500
    else:
        num_files = max_files / 2
    num_files = int(num_files)   
    X = numpy.zeros(((2 * num_files), len(vocab)))
    Y = numpy.zeros((2 * num_files))
         
    i = 0
    j = 0
    while i < int(num_files):
        file = open(pos_dir + pos_filenames[i])
        feat_vec = create_word_vector(file, vocab)  
        X[j] = feat_vec
        Y[j] = 1
        i += 1
        j += 1
        
    i = 0
    while i < int(num_files):
        file = open(neg_dir + neg_filenames[i])
        feat_vec = create_word_vector(file, vocab)  
        X[j] = feat_vec
        i += 1
        j += 1
       

    return X, Y