# coding:utf8
import numpy as np

from sklearn.datasets.samples_generator import make_classification
total_words = 0

# 读text文件,支持多文件一起读，将token转换为index，并返回两者的dict和转换为index的sentence
def read_texts(text_file_list, start_line=0):
    global total_words
    total_words = 0
    if type(text_file_list) != type([]):
        text_file_list = [text_file_list]
    token_id = {}
    sentences = []
    index = 0
    count = -1
    for text_file in text_file_list:
        for line in open(text_file):
            count += 1
            if count < start_line:
                continue
            line = line.strip().split(' ')
            sentence = []
            for token in line:
                if token in token_id:
                    sentence.append(token_id[token])
                else:
                    token_id[token] = index
                    sentence.append(index)
                    index += 1
            sentences.append(sentence)
            total_words += len(sentence)
    return token_id, sentences

# 使用scikit-learn的函数初始化，或者用普通的随机初始化word embeding
def init_embedding(token_id, dimension=400, sclar=1., standard_normal=True):
    if standard_normal:
        return sclar * np.random.rand(len(token_id), dimension)
    else:
        X, y = make_classification(n_samples=len(token_id), n_features=dimension, n_redundant=0, n_informative=dimension, n_classes=min(int(np.sqrt(len(token_id) / 2)), dimension), n_clusters_per_class=1, random_state=5756239, shuffle=True)
        return sclar * X

# 任意格式字符串转换成utf8格式
def any2utf8(text, errors='strict', encoding='utf8'):
    if isinstance(text, unicode):
        return text.encode('utf8')
    return unicode(text, encoding, errors=errors).encode('utf8')

# 保存word embedding到文件
def save_embedding(save_file, word_embedding, token_id, binary=False):
    writer = open(save_file, 'w')
    writer.write(any2utf8("%s %s\n" % word_embedding.shape))
    fake_vector = np.random.rand(len(word_embedding[0]))
    if binary:
        writer.write(any2utf8('</s>') + b" " + fake_vector.tostring())
    else:
        writer.write(any2utf8("%s %s\n" % ('</s>', ' '.join("%f" % val for val in fake_vector))))
    for target in token_id:
        if type(target) == type(''):
            if binary:
                writer.write(any2utf8(target) + b" " + word_embedding[token_id[target]].tostring())
            else:
                writer.write(any2utf8("%s %s\n" % (target, ' '.join("%f" % val for val in word_embedding[token_id[target]]))))
    writer.close()

# 使用最大间隔思想训练word embedding
def skip_gram_train(word_embedding, sentences, token_id, window_size=5, lr=0.025, epoch=1, margin=1, penal_scale=1, file_prefix=None, auto_save=False, save_frequency=20, binary=False):
    print 'total words:', total_words
    full_percent = 100
    percentage = [int(i / 100.*total_words) for i in range(full_percent)]
    percent = 0    
    W = np.random.randn(word_embedding.shape[1], len(token_id))
    word_count = 0
    last_word_count = 0
    starting_lr = lr
    min_lr = 0.0001 * starting_lr
    saved = False
    for ep in range(epoch):
        print 'epoch:', ep + 1
        if auto_save and file_prefix and (ep + 1) % save_frequency == 0:
            save_embedding(file_prefix + 'Ep' + str(ep + 1) + '.bin', word_embedding, token_id, binary=binary)
            saved = True
        for sentence in sentences:
            for i in range(len(sentence)):
                x_index = sentence[i]
                for j in range(max(i - window_size, 0), i) + range(i + 1, min(i + window_size + 1, len(sentence))):
                    y = sentence[j]
                    x = word_embedding[x_index]
                    w = W[:, y]
                    if np.dot(w, x) < margin:  # 默认间隔为1
                        W[:, y] += lr * x  # (x - lmd * W[:, y])
                        word_embedding[x_index] += lr * w
                word_count += 1
                # 与mikolov的学习率自适应方式保持一致
                if word_count % 1000 == 0 and word_count - last_word_count > 10000:
                    lr = max(starting_lr * (1 - word_count / (total_words + 1.0)), min_lr)
                while percent < len(percentage) and word_count > percentage[percent]:
                    percent += 1
                    print percent, '%, learning rate:', lr 
        if auto_save and not saved and ep == 0:
            save_embedding(file_prefix + 'Ep' + str(ep + 1) + '.bin', word_embedding, token_id, binary=binary)
            saved = True
    if auto_save and not saved:
        save_embedding(file_prefix + 'Ep0.bin', word_embedding, token_id, binary=binary)
    return word_embedding

def get_target_words_id(file_name, token_id):
    target_words = set()
    for line in open(file_name):
        for word in line.strip().split(' '):
            target_words.add(word)
    target_words_id = {}
    index = 0
    target_ids = set()
    for w in target_words:
        target_words_id[w] = index
        target_words_id[token_id[w]] = index
        target_ids.add(token_id[w])
        index += 1
    return target_words_id, target_ids


def train_MMWR(text_file, save_file, dimension=400, binary=False):
    token_id, sentences = read_texts(text_file)
    word_embedding = init_embedding(token_id , dimension=dimension)
    word_embedding = skip_gram_train(word_embedding, sentences, token_id)
    save_embedding(save_file, word_embedding, token_id, binary=binary)

if __name__ == '__main__': 
    train_MMWR('forWord2vec_PPI_10t', 'MMWR_d400_epoch1.bin', dimension=400, binary=False)

      
