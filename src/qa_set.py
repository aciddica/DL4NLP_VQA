import numpy
import utils
import pandas as pd
import re
from copy import deepcopy
import word2vec
class QASet:
    def __init__(self, part):
        # 加载指定问题答案集合
        self.varilen = True
        self.Q_set,self.A_set = utils.load_json_file(path = '../data', mode = part, cleaned=False)
        # 问题格式处理，删除标点符号、分词等操作
        self.normalize()
        self.Q_A_df = self.QAmerge()
        # 加载词向量模型
        self.word2vec_model = word2vec.load_word2vec('word_vec/embedding.txt')
        
        qst_vec = []
        for item in self.Q_A_df['question']:
            qst_vec.append(word2vec.QA2ndarray(item,self.word2vec_model))

        self.Q_A_df['qst_vec'] = qst_vec

        ans_vec = []
        for item in  self.Q_A_df['annotation']:
            ans_vec.append(word2vec.QA2ndarray(item,self.word2vec_model))
        
        self.Q_A_df['ans_vec'] = ans_vec
        self.Q_A_df.head()

    def normalize(self):
        for item in self.Q_set:
            qst = item['question']
            qst = qst.lower()
            qst = re.sub(r"[^a-zA-Z]+", r" ",qst)
            qst = qst.strip()
            qst_vocab = qst.split(' ')
            if not self.varilen:
                qst_vocab = utils.sentence_align(qst_vocab,8)
            item['question'] = deepcopy(qst_vocab)
        
        for item in self.A_set:
            ans = item['multiple_choice_answer']
            ans = ans.lower()
            ans = re.sub(r"[^a-zA-Z]+", r" ",ans)
            ans = ans.strip()
            ans_vocab = ans.split(' ')
            if not self.varilen:
                ans_vocab = utils.sentence_align(ans_vocab,8)
            item['multiple_choice_answer'] = deepcopy(ans_vocab)

    def QAmerge(self):
        Q_df = pd.DataFrame(self.Q_set)
        A_df = pd.DataFrame(self.A_set)
        Q_A_df = Q_df.merge(A_df,on = ['question_id','image_id'])
        Q_A_df = Q_A_df[['question_id','image_id','question','multiple_choice_answer']]
        Q_A_df['annotation'] = Q_A_df['multiple_choice_answer']
        Q_A_df = Q_A_df[['question_id','image_id','question','annotation']]
        Q_A_df.head()
        return Q_A_df
        
    def __getitem__(self, index):
        return self.Q_A_df['qst_vec'][index], self.Q_A_df['ans_vec'][index], self.Q_A_df['image_id'][index]
    
    def __len__(self):
        return len(self.Q_A_df)
