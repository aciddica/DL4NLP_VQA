import numpy
import os
import re
import mindspore
from multiprocessing import cpu_count
try:
    import gensim
except:
    os.system('pip install gensim')
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim import corpora

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
    model = Word2Vec(Vocab,vector_size=100,window = 3,min_count = 1,workers = cpu_count(),sg = 1)
    model.wv.save_word2vec_format(path,binary=False)
    return model

def load_word2vec(path):
    return KeyedVectors.load_word2vec_format(path)

def QA2ndarray(qst,model):
    qst_tensor = []
    for item in qst:
        if item == '<INS>':
            qst_tensor.append(numpy.array([0 for _ in range(100)],dtype=numpy.float32))
        else :
            qst_tensor.append(model[item])
            if len(model[item]) != 100:
                raise Exception("lenth fail : {},word : {}".format(len(model[item]),item))
    return numpy.array(qst_tensor)

# def ans2ndarray(ans, model):

    
def other(Vocab):
    '''
    just put some 'may be useful' code there
    '''
    dictionary = corpora.Dictionary(Vocab)   # 生成词典
    dictionary.save('./pre/vocab.dict')  # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(sentence) for sentence in Vocab]
    corpora.MmCorpus.serialize('./pre/vocab.mm', corpus)

'''
TODO
fill in the functions below
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
def decode_embedding(prediction):
    '''
    prediction: mindspore.Tensor in shape (size_batch, length_output_vector)
    this is the raw output of a VQANet
    decode each row to its embedding, and return mindspore.Tensor in shape (size_batch, 8, 100)
    e.g.
    prediction = mindspore.Tensor([[0.2, 2.5, ...], [-3.1, 0, ...]])
    decode(prediction) -> mindspore.Tensor(
        [
            [[1.1, ...], [...], [...], [...], [...], [...], [...], [...]], # answer 1
            ... # answer 2
        ],
        mindspore.float32
    )
    '''
    # dummy code
    return mindspore.ops.Zeros()((prediction.shape[0], 8, 100), mindspore.float32)
def loss(prediction, answer):
    '''
    prediction: mindspore.Tensor in shape (size_batch, length_output_vector)
    this is the raw output of a VQANet
    answer: mindspore.Tensor in shape ((size_batch,) + shape_answer)
    this is the annotation provided by a VQASet
    calculate the loss between prediction & answer, and assign to loss
    this function will be differentiated, so do not incorporate complicated algorithms like for-clauses
    '''
    # dummy code
    return mindspore.ops.Zeros()((), mindspore.float32)
