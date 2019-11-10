
import pandas as pd
import os,re
import jieba
import copy
import codecs
from gensim.models import Word2Vec
import numpy as np

def seg_line(line):
    tokens = jieba.cut(str(line), cut_all=False)
    words = []
    for word in tokens:
        words.append(word)
    return " ".join(words)

# 清理数据
def split_data(train_path,
               test_path,
               train_save_path,
               test_save_path
               ):

    train = pd.read_csv(train_path,encoding='utf-8')
    test = pd.read_csv(test_path,encoding='utf-8')

#     去除空值
    train.dropna(axis=0,how="any",inplace=True)
    test.dropna(axis=0, how="any", inplace=True)

#  重新设置索引
    train.reset_index(inplace=True,drop=True)
    test.reset_index(inplace=True, drop=True)

# 在循环中进行处理

    for k in ['Brand','Model','Question','Dialogue','Report']:
        for i in range(len(train[k])):
            line = train[k].get(i)
            line = re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+', str(line))
            line = ''.join(line)
            train[k][i] = seg_line(line)
    for k in ['Brand','Model','Question','Dialogue']:
        for i in range(len(test[k])):
            if pd.isnull(test[k][i]):
                print('get one')
                continue
            line = test[k].loc[i]
            line = re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+', str(line))
            line = ''.join(line)
            test[k][i] = seg_line(line)
    train.dropna(axis=0, how='any', inplace=True)
    test.dropna(axis=0, how='any', inplace=True)

 #这里再次去除空值是因为有些字段太少，再去重停用词阶段被消除了，所以还要去空值
    train['input'] = train['Brand'] + ' ' + train['Model'] + ' ' + train['Question'] + ' ' + train['Dialogue']
    test['input'] = test['Brand'] + ' ' + test['Model'] + ' ' + test['Question'] + ' ' + test['Dialogue']

    train.drop(['Brand','Model','Question','Dialogue'],axis=1,inplace=True)
    test.drop(['Brand', 'Model', 'Question','Dialogue'], axis=1,inplace=True)

    train.to_csv(train_save_path,index=False,encoding='utf-8')
    test.to_csv(test_save_path,index=False,encoding='utf-8')

def stat_dict(lines):
    word_dict = {}
    for line in lines:
        tokens = str(line).split(" ")
        for t in tokens:
            t = t.strip()
            if t:
                word_dict[t] = word_dict.get(t,0) + 1
    return word_dict

def filter_dict(word_dict, min_count=3):
    out_dict = {}
    keys = word_dict.keys()
    for k in keys:
        if word_dict[k] >= min_count:
            out_dict[k] = word_dict[k]
    return out_dict


def build_vocab(lines, min_count=3):
    start_token = u"<s>"
    end_token = u"<e>"
    unk_token = u"<unk>"
    word_dict = stat_dict(lines)
    word_dict = filter_dict(word_dict, min_count)
    sorted_dict = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
    sorted_words = [w for w,c in sorted_dict]
    sorted_words = [start_token, end_token, unk_token] + sorted_words
    vocab = dict([(w,i) for i,w in enumerate(sorted_words)])
    reverse_vocab = dict([(i,w) for i,w in enumerate(sorted_words)])
    return vocab, reverse_vocab


def save_vocab(vocab, path):
    output = codecs.open(path, "w", "utf-8")
    for w,i in sorted(vocab.items(), key=lambda x:x[1]):
        output.write("%s %d\n" %(w,i))
    output.close()

def load_vocab(path):
    vocab = {}
    input = codecs.open(path, "r", "utf-8")
    lines = input.readlines()
    input.close()
    for l in lines:
        w, c = l.strip().split(" ")
        vocab[w] = int(c)
    return vocab

# 本程序的作用是进行数据预处理，包括去空值，去重等
if __name__ == '__main__':
    from config import config
    train_path = config.train_source_path
    test_path = config.test_source_path

    train_save_path = config.train_save_path
    test_save_path = config.test_save_path

    vocab_path = config.vocabulary_path

    print('start to preprocess the source data')

    split_data(train_path,
               test_path,
               train_save_path,
               test_save_path
               )

    print('end of the preprocess')

    min_count = 1

    train = pd.read_csv(train_save_path, encoding='utf-8')

    lines = []
    for k in ['input', 'Report']:
        lines.extend(list(train[k].values))

    vocab, reverse_vocab = build_vocab(lines, min_count)

    print('save vocabulary begin')
    save_vocab(vocab, vocab_path)
    print('save vocabulary end')

