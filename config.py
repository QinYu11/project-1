#－－－－－－－－－－－－　　路径配置　　－－－－－－－－

import os
# 项目路径
project_path ='/home/qinyu/Desktop/projectone/dataset/'


# 原始训练数据路径
# train_source_path = '/home/qinyu/Desktop/projectone/dataset/AutoMaster_TrainSet.csv'
train_source_path = os.path.join(project_path,'AutoMaster_TrainSet.csv')

#　原始测试数据路径
# test_source_path = '/home/qinyu/Desktop/projectone/dataset/AutoMaster_TestSet.csv'
test_source_path = os.path.join(project_path,'AutoMaster_TestSet.csv')

#　停用词路径
# stop_words_path = '/home/qinyu/Desktop/projectone/dataset/stop_words.txt'
stop_words_path = os.path.join(project_path,'stop_words.txt')

#　切词后训练文本路径
# train_result_path = '/home/qinyu/Desktop/projectone/dataset/AutoMaster_Train_result_Set.txt'
train_result_path = os.path.join(project_path,'AutoMaster_Train_result_Set.txt')

#　切词后测试文本路径
# test_result_path = '/home/qinyu/Desktop/projectone/dataset/AutoMaster_Test_result_Set.txt'
test_result_path = os.path.join(project_path,'AutoMaster_Test_result_Set.txt')

#　训练后词向量存放路径
# model_path = '/home/qinyu/Desktop/projectone/dataset/w2v.model'
model_path = os.path.join(project_path,'w2v.model')

#　整理后训练数据
# train_save_path = '/home/qinyu/Desktop/projectone/dataset/train.csv'
train_save_path = os.path.join(project_path,'train.csv')

#　整理后测试数据
# test_save_path = '/home/qinyu/Desktop/projectone/dataset/test.csv'
test_save_path = os.path.join(project_path,'test.csv')

# 词典路径
# vocabulary_path = '/home/qinyu/Desktop/projectone/dataset/vocabulary.txt'
vocabulary_path = os.path.join(project_path,'vocabulary.txt')

# 训练后模型存放位置
model_save_path = os.path.join(project_path,'model_save')

# 为了区分是在笔记本上面训练还是在服务器上面训练，设置一个运行环境参数
# 1 代表是在笔记本上面进行训练，所以需要少一些的训练数据
# 2 代码是在服务进行训练，不需要限制数据量
run_environment = '1'

# 当是在笔记本上面进行训练的时候，设置一个限制值
max_train_data = 1000

#　其他训练参数
max_features = 300
maxlen = 300
embed_size = 100
max_length_inp = 500
max_length_targ=  50





