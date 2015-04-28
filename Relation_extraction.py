# coding:utf8
import conf
import numpy as np
import random
from Skeleton_features import get_features
from Vector_composition import get_input
import pickle
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression

clf = None
word2vec = None

def load_word2vec(word2vec_file=conf.word2vec_file):
    global word2vec
    if word2vec is None:
        pad_vector = None
        word2vec = {}
        firstLine = True
        for line in open(word2vec_file):
            if firstLine and line.count(' ') < 50:
                firstLine = False
                continue
            line = line.strip()
            sep = line.find(' ')
            token = line[:sep]
            vector = [float(v) for v in line[sep + 1:].split(' ')]
            if pad_vector is None:
                pad_vector = np.zeros_like(vector)
                word2vec[conf.pad] = pad_vector
            if token in word2vec:
                word2vec[token] = np.concatenate([word2vec[token], vector])
            else:
                word2vec[token] = np.asarray(vector)
    return word2vec

def get_labels(instances):
    Y = []
    for record in instances:
        Y.append(record[0] == 1 and 1 or 0)
    return Y    
    
def get_data(corpora_pkl, word2vec_file):
    word2vec = load_word2vec(word2vec_file)
    instances = pickle.load(open(corpora_pkl))
    random.seed(2)
    random.shuffle(instances)
    input_features = get_features(instances)
    X = np.asarray(get_input(input_features, word2vec))
    Y = np.asarray(get_labels(instances))
    return X, Y

def validate(X, Y):
    global clf
    clf = LogisticRegression(C=conf.C, class_weight='auto', penalty='l1')
    scores = cross_validation.cross_val_score(clf, X, Y, cv=conf.cv, scoring='f1', pre_dispatch=4)  # scoring='roc_auc',
    return scores
    
def validate_all():
    for corpora in [conf.aimed_pkl, conf.bioinfer_pkl, conf.hprd50_pkl, conf.iepa_pkl, conf.lll_pkl]: 
        X, Y = get_data(corpora, conf.word2vec_file)
        print 'data ready'
        scores = validate(X, Y)
        print scores
        print sum(scores) / conf.cv
    

if __name__ == '__main__':
    validate_all()
    
    
    
    
