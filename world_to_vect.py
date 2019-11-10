import jieba
import os
import time
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

def get_stop_words(file: str, encoding='utf-8'):
    ret = [l for l in open(file, encoding=encoding).read()]
    return ret

def write_token_to_file(infile, outfile):
    words = []
    for line in open(infile, 'r', encoding='utf-8'):
        line = line.strip()
        if line:
            w = jieba.lcut(line)
            words += w + ['\n']
    outfile.writelines(' '.join(words))


def train_w2v_model(txtPath,model_path):
    start_time = time.time()
    # 训练词向量
    w2v_model = Word2Vec(LineSentence(txtPath), workers=4)  # Using 4 threads
    # 保存词向量
    w2v_model.save(model_path) # Can be used for continue trainning
    # w2v_model.wv.save('w2v_gensim') # Smaller and faster but can't be trained later
    print('elapsed time:', time.time() - start_time)


def get_model_from_file(model_path):
    # model = KeyedVectors.load('w2v_gensim', mmap='r')
    model = Word2Vec.load(model_path)
    return model


# 本程序的作用是根据第一步生成的文本文件，创建词向量文件
if __name__ == "__main__":

    from config import config

# 文本路径　（文本是前面一步，切词的结果）
    txt_path =config.train_result_path
# 保存词向量路径
    model_path=config.model_path

    # 调用Gensim 包处理单词到词向量的转换，并且保存起来
    print('start to convert worlds to vector')
    train_w2v_model(txt_path,model_path)

    model = get_model_from_file(model_path)
    words_list = model.vocabulary

    print('total words:')
    print(model.corpus_total_words)
    print('world list')
    print(words_list)
    print('end of this sub program')