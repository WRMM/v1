# coding:utf8
from End.TCBB import conf

def decode_offset(offset):
    rawOffsets = offset.split('-')
    return [int(rawOffsets[0]), int(rawOffsets[1])]

def convert_offsets(space_indices, offset_str_list):
    if offset_str_list is None:
        return None
    offset_list = []
    for offset_str in offset_str_list:
        offset = decode_offset(offset_str)
        for index in space_indices:
            if index <= offset[0]:
                offset[0] += 1
            if index <= offset[1]:
                offset[1] += 1
        offset_list.append(offset)
    return offset_list

def get_index(sentence, offset):
    start = sentence[:offset[0]].count(' ')
    end = sentence[offset[0]:offset[1]].count(' ') + start
    return range(start, end + 1)


def tokenize_sentence_i(sentence, offset_list=None, capital_lower=False, index_format = True):
    new_sent = ''
    first_one = True
    if not capital_lower:
        first_one = False
    new_space_indices = []
    index = 0
    for token in sentence.split(' '):
        new_token = ''
        for i in range(len(token)):
            cc = token[i]
            lc = ''
            if len(new_token) != 0:
                lc = new_token[-1]
            if cc in conf.special_charactors:
                if i > 0 and lc != ' ':
                    new_token += ' '
                    new_space_indices.append(index)
                    index += 1
                new_token += token[i]
                index += 1
                continue
            if lc in conf.special_charactors and cc != ' ':
                new_token += ' '
                new_space_indices.append(index)
                index += 1
            new_token += token[i]
            index += 1
        if first_one:
            new_token = capital_lower(new_token)
            first_one = False
        new_sent += new_token + ' '
        index += 1
    new_sent = new_sent[:-1]
    if index_format:
        return new_sent, [get_index(new_sent, offset) for offset in convert_offsets(new_space_indices, offset_list)]
    else:
        return new_sent, convert_offsets(new_space_indices, offset_list)


