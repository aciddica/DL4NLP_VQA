import json
import numpy as np
import re
import os

def load_json_file(path,mode,cleaned):
    '''
    path : 到data的目录,包含data,不包含末尾'/'
    return : list,每个entry都是一个字典
            Q_set : {}
            A_set : {}
    '''
    # j_eval = open(path + '/questions/val.json')
    # j_train = open(path + '/questions/train.json')
    # j_test = open(path+'/questions/test.json')
    if(mode == 'train'):
        j_path = open(os.path.join(path, 'question/train.json'))
    elif(mode == 'val'):
        j_path = open(os.path.join(path, 'question/val.json'))
    else:
        j_path = open(os.path.join(path, 'question/test.json'))
    if cleaned:
        Q_set = json.load(j_path)
    else:
       Q_set = json.load(j_path)['questions']
    j_path.close()
    # j_eval.close()
    # j_train.close()
    # j_train.close()

    # j_eval = open(path +'/annotations/val.json')
    # j_train = open(path + '/annotations/train.json')
    # j_test = open(path + '/annotations/test.json')
    if(mode == 'train'):
        a_path = open(os.path.join(path, 'question/train.json'))
    elif(mode == 'val'):
        a_path = open(os.path.join(path, 'question/val.json'))
    else:
        a_path = open(os.path.join(path, 'question/test.json'))
    if cleaned:
        A_set = json.load(a_path)
    else:
        A_set = json.load(a_path)['annotations']

    a_path.close()
    print(mode," : {}".format(len(A_set)))
    return Q_set,A_set

def get_all_img_id(path):
    '''
    返回有重复的id和全部的id
    不要排序
    '''
    path_train = path + '\train'
    path_eval = path + '\val'
    path_test = path + '\test'
    train_img = os.listdir(path_train)
    eval_img = os.listdir(path_eval)
    test_img = os.listdir(path_test)
    img_id = []
    for item in train_img+eval_img+test_img:
        unit = item.split('_')
        id = unit[2]
        id = id.split('.')[0]
        while id[0]=='0':
            id=id[1:]
        img_id.append(int(id))
    return img_id

def get_question_img_id(Q_set):
    '''
    fuction : 返回Q_set里面所有question涉及到的img_id
    '''
    Q_img_id = []
    for item in Q_set:
        id = item['image_id']
        Q_img_id.append(id)
    return Q_img_id

def get_answer_img_id(A_set):
    '''
    fuction : 返回A_set里面所有annotation涉及到的img_id
    '''
    A_img_id = []
    for item in A_set:
        id = item['image_id']
        # if id not in Q_img_id:
        A_img_id.append(id)
    return A_img_id

def get_answer_Q_id(A_set):
    A_Q_id = []
    for item in A_set:
        id = item['question_id']
        A_Q_id.append(id)
    return A_Q_id

def get_Q_id(Q_set):
    Q_id = []
    for item in Q_set:
        id = item['question_id']
        # if id not in Q_img_id:
        Q_id.append(id)
    return Q_id


def check_QA_dataset_for_img(Q_img_id,img_id):
    '''
    返回没用的问题的索引（数组下角标）
    返回没用的图片id
    '''
    index = []
    img_nowhere = []
    # img_nouse = []
    for i,item in enumerate(Q_img_id):
        if item not in img_id:
            index.append(i)
            img_nowhere.append(item)

    print("{} questions/anwser have no corrsponding img .{}".format(len(img_nowhere),len(img_nowhere)/len(Q_img_id)))
    return index

def check_QA_corres(Q_id,A_Q_id):
    '''
    检查answer里面是不是回答了不存在的问题
    检查是不是有的Q没有A回答
    '''
    Q_index = []
    A_index = []
    for i,item in enumerate(Q_id):
        '''
        是不是有的问题没有人回答
        '''
        if item not in A_Q_id:
            Q_index.append(i)
    
    for i,item in enumerate(A_Q_id):
        if item not in Q_id:
            A_index.append(i)
    
    return Q_index,A_index

def gen_clean_dataset(Q_set,A_set,Q_index,A_index,path):
    train_num = 44375
    eval_num = 21435
    test_num = 21435

    # train set
    train_Q_set = []
    eval_Q_set = []
    test_Q_set = []

    for i,item in enumerate(Q_set):
        if i not in Q_index:
            if i < train_num:
                train_Q_set.append(item)
            elif i < train_num+eval_num:
                eval_Q_set.append(item)
            else:
                test_Q_set.append(item)
    
    # path = "./clean_data/questions/"
    train_name = "/questions/train.json"
    eval_name = "/questions/val.json"
    test_name = "/questions/test.json"

    file1 = open(path+train_name,mode='w')
    file2 = open(path+eval_name,mode='w')
    file3 = open(path+test_name,mode='w')

    json.dump(train_Q_set,file1)
    json.dump(eval_Q_set,file2)
    json.dump(test_Q_set,file3)

    file1.close()
    file2.close()
    file3.close()

    train_A_set = []
    eval_A_set = []
    test_A_set = []

    for i,item in enumerate(A_set):
        if i not in A_index:
            if i < train_num:
                train_A_set.append(item)
            elif i < train_num+eval_num:
                eval_A_set.append(item)
            else:
                test_A_set.append(item)

    # path = "./clean_data/annotations/"
    train_name = "/annotations/train.json"
    eval_name = "/annotations/val.json"
    test_name = "/annotations/test.json"

    file11 = open(path+train_name,mode='w')
    file12 = open(path+eval_name,mode='w')
    file13 = open(path+test_name,mode='w')

    json.dump(train_A_set,file11)
    json.dump(eval_A_set,file12)
    json.dump(test_A_set,file13)

    file11.close()
    file12.close()
    file13.close()

def sentence_align(Q,length):
    if len(Q) == length :
        return Q
    elif len(Q) < length:
        while len(Q) < length:
            # 句子传入的时候检查单词是否==《INS》
            # 如果是填充符，那么对应词的词向量默认做成[0*100]
            Q.append("<INS>")
    elif len(Q) > length:
        x = len(Q)
        for _ in range(x-length):
            Q.pop(len(Q)-1)
    
    return Q