import numpy
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

# def qst2tenser(qst,model):
#     '''
#     note: 有bug
#     '''
#     qst_tensor = []
#     for item in qst:
#         if item == '<INS>':
#             qst_tensor.append([[0]*100])
#         else :
#             qst_tensor.append(model[item])
#     return qst_tensor
def qst2ndarray(qst,model):
    return numpy.array([model[i] for i in qst if i != '<INS>'], numpy.float32)
    
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

'''
TODO
fill in the functions below
'''
def annotation2ndarray(annotation, model):
    '''
    annotation: element in Q_A_df['annotation']
    returns numpy.ndarray of dtype numpy.float32
    '''
def decode(prediction):
    '''
    prediction: mindspore.Tensor in shape (size_batch, length_output_vector)
    this is the raw output of a VQANet
    decode each row to a str, and return a tuple of strs
    e.g.
    prediction = mindspore.Tensor([[0.2, 2.5, ...], [-3.1, 0, ...]])
    decode(prediction) -> 'yes', 'two'
    '''
def accuracy(prediction, answer):
    '''
    prediction: mindspore.Tensor in shape (size_batch, length_output_vector)
    this is the raw output of a VQANet
    answer: mindspore.Tensor in shape ((size_batch,) + shape_answer)
    this is the annotation provided by a VQASet
    returns the proportion of correct predictions
    e.g.
    prediction = mindspore.Tensor([
        [0.2, 2.5, ...], # correct
        [-3.1, 0, ...], # wrong
    ])
    answer = mindspore.Tensor([annotation1, annotation2])
    accuracy(prediction, answer) -> 0.5
    '''
def loss(prediction, answer):
    '''
    prediction: mindspore.Tensor in shape (size_batch, length_output_vector)
    this is the raw output of a VQANet
    answer: mindspore.Tensor in shape ((size_batch,) + shape_answer)
    this is the annotation provided by a VQASet
    calculate the loss between prediction & answer, and assign to loss
    this function will be differentiated, so do not incorporate complicated algorithms like for-clauses
    '''
