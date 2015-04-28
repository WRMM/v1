# coding:utf8
import numpy as np
import conf

def digitalize(feature, word2vec, alpha=None, trunc=False):
    mat = None
    if alpha is not None:
        mat = [alpha * word2vec[v] if v in word2vec else alpha * word2vec[conf.pad] for v in feature]
    else:
        mat = [word2vec[v] if v in word2vec else word2vec[conf.pad] for v in feature ]
    x = None
    if trunc:
        x = [np.concatenate(mat)]
    else:
        x = [np.max(mat, axis=0)]
    return x
       

def get_input(input_features, word2vec):
    if type(word2vec) == type(''):
        from End.Relation_extraction import load_word2vec
        word2vec = load_word2vec()
    input_matrix = []
    for entity1, entity2, context_words, inner_words, tokens in input_features:
        x_entity1, x_entity2, x_context_words, x_inner_words, x_tokens = digitalize(entity1, word2vec, 0.5), digitalize(entity2, word2vec, 0.5), digitalize(context_words, word2vec, trunc=True), digitalize(inner_words, word2vec), digitalize(tokens, word2vec, 0.5)
        input_vector = np.concatenate(x_entity1 + x_entity2 + x_context_words + x_inner_words + x_tokens)
        input_matrix.append(input_vector)
    return input_matrix
        
    
