import jieba

def pre_process_csv( train_path,
                     test_path,
                     stop_words_path,
                     train_result_path,
                     test_result_path):

    with open(train_path, 'r', encoding='utf-8') as f_read:
            data1 = f_read.readlines()
    f_read.close()

    with open(test_path,'r',encoding='utf-8') as f_read2:
            data2 = f_read2.readlines()
    f_read2.close()

    stopwords = [word.strip() for word in open(stop_words_path,'r',encoding='utf-8').readlines()]

    with open(train_result_path,'w',encoding='utf-8') as f_in:
        temp1 = []
        for line in data1:
                    line = list(jieba.cut(str(line).strip()))
                    for word in line:
                        # if word not in stopwords:
                        if word in stopwords:
                            continue
                        temp1.append(word)
        f_in.write(' '.join(temp1))
        print('train data write sucessfully')
        temp2 = []

        for line in data2:
                    line = list(jieba.cut(str(line).strip()))
                    for word in line:
                        # if word not in stopwords:
                        if word in stopwords:
                            continue
                        temp2.append(word)
        f_in.write(' '.join(temp2))
        f_in.close()
        print('test data write sucessfully')

    ev_sucess = 'Y'

    return ev_sucess

# 本子程序的作用是把原始数据进行切词处理，处理结果是生成切词后的文本文件
if __name__ == '__main__':
    msg_sucess = ''
    from config import config
    print('start to process')
    msg_sucess = pre_process_csv(config.train_source_path,
                                 config.test_source_path,
                                 config.stop_words_path,
                                 config.train_result_path,
                                 config.test_result_path )
    print(msg_sucess)
    print('end of process')








    













