# coding:utf8
import conf


def get_feature(sentence, p1range, p2range, r=conf.r, pad=conf.pad):
    tokens = sentence.split(' ')
    context_words = []
    for i in range(min(p1range) - r + 1, min(p1range)) + range(max(p1range) + 1, max(p1range) + r) + range(min(p2range) - r + 1, min(p2range)) + range(max(p2range) + 1, max(p2range) + r):
        if i < 0 or i > len(tokens) - 1:
            context_words.append(pad)
        else:
            context_words.append(tokens[i])

    inner_words = []
    for i in range(max(p1range) + 1, min(p2range)):
        inner_words.append(tokens[i])
    if len(inner_words) == 0:
        inner_words.append(pad)
    
    entity1 = []
    for i in p1range:
        entity1.append(tokens[i])

    entity2 = []
    for i in p2range:
        entity2.append(tokens[i])
        
    return entity1, entity2, context_words, inner_words, tokens

def get_features(instances, r=conf.r, pad=conf.pad):
    input_feature = []
    for target, sentence, p1range, p2range in instances:
        entity1, entity2, context_words, inner_words, tokens = get_feature(sentence, p1range, p2range, r, pad)
        input_feature.append([entity1, entity2, context_words, inner_words, tokens])
    return input_feature
