import re
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim import corpora
from collections import defaultdict

def get_vocab(Q_set,A_set,config,stoplist):
    
    Vocab = []
    for i in range(len(Q_set)):
        Q = Q_set[i]['question']
        A = A_set[i]['multiple_choice_answer']
        Q = Q.lower()
        Q = re.sub(r"[^a-zA-Z]+", r" ",Q)
        Q = Q.strip()
        A = A.lower()
        A = re.sub(r"[^a-zA-Z]+", r" ",A)
        A = A.strip()

        Q_vocab = Q.split(' ')
        if(config['use_stopwords']):
            Vocab.append([word for word in Q_vocab if word not in stoplist])
        else:
            Vocab.append(Q_vocab)

        A_vocab = A.split(' ')
        if(config['use_stopwords']):
            Vocab.append([word for word in A_vocab if word not in stoplist])
        else:
            Vocab.append(A_vocab)

    return Vocab

def gen_word2vec(Vocab,path):
    from multiprocessing import cpu_count
    model = Word2Vec(Vocab,vector_size=100,window = 3,min_count = 1,workers = cpu_count(),sg = 1)
    model.wv.save_word2vec_format(path,binary=False)
    return model

def load_word2vec(path):
    return KeyedVectors.load_word2vec_format(path)

def qst2tenser(qst,model):
    '''
    note: 尚未测试，可能有bug
    '''
    qst_tensor = []
    for item in qst:
        if item == '<INS>':
            qst_tensor.append([[0]*100])
        else :
            qst_tensor.append(model[item])
    return qst_tensor
    
def other(Vocab):
    '''
    just put some 'may be useful' code there
    '''
    dictionary = corpora.Dictionary(Vocab)   # 生成词典

    # 将文档存入字典，字典有很多功能，比如
    # diction.token2id 存放的是单词-id key-value对
    # diction.dfs 存放的是单词的出现频率
    dictionary.save('./pre/vocab.dict')  # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(sentence) for sentence in Vocab]
    corpora.MmCorpus.serialize('./pre/vocab.mm', corpus)
