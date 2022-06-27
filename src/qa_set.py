import numpy
import utils
import pandas as pd
import re
from copy import deepcopy
import word2vec
class QASet:
    def __init__(self, part):
        self.Q_set,self.A_set = utils.load_json_file(path = './data', mode = part, cleaned=False)
        for item in self.Q_set:
            qst = item['question']
            qst = qst.lower()
            qst = re.sub(r"[^a-zA-Z]+", r" ",qst)
            qst = qst.strip()
            qst_vocab = qst.split(' ')
            qst_vocab = utils.sentence_align(qst_vocab,8)
            item['question'] = deepcopy(qst_vocab)
        
        self.Q_df = pd.DataFrame(self.Q_set)
        self.A_df = pd.DataFrame(self.A_set)

        self.Q_A_df = self.Q_df.merge(self.A_df,on = ['question_id','image_id'])
        self.Q_A_df = self.Q_A_df[['question_id','image_id','question','multiple_choice_answer']]
        self.Q_A_df['annotation'] = self.Q_A_df['multiple_choice_answer']
        self.Q_A_df = self.Q_A_df[['question_id','image_id','question','annotation']]
        self.Q_A_df.head()
        self.word2vec_model = word2vec.load_word2vec('word_vec/embedding.txt')
        
        rst = []
        for item in  self.Q_A_df['question']:
            rst.append(word2vec.qst2ndarray(item,self.word2vec_model))

        self.Q_A_df['question'] = rst
        rst = []
        for item in  self.Q_A_df['annotation']:
            rst.append(word2vec.annotation2ndarray(item,self.word2vec_model))

        self.Q_A_df['annotation'] = rst
        self.Q_A_df.head()

        # word2vec_model
        
    def __getitem__(self, index):
        return self.Q_A_df['question'][index], self.Q_A_df['question'][index], self.Q_A_df['annotation'][index]
    
    def __len__(self):
        return len(self.Q_A_df)
