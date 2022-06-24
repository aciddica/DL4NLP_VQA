- **如果要跑代码，clean_QAdataset.ipynb、QA.ipynb应该和data文件夹处于同一个父目录下**

- clean_QAdataset : 对原本数据集做处理、删除对应的img_id不存在的问题和答案
    - 需要的目录结构：
        --data
        -----questions
        -----annotations
        -----img
        --clean_QAdataset.ipynb
    - 不需要作上述处理直接不运行这个ipynb即可

- QA.ipynb : 处理vocab、训练词向量用的代码
    - vocab就是所有问题和答案中出现的单词
    - 调用gensim的word2wec库
    - 词向量已经训练完毕并保存，load的代码参见QA.ipynb的第23\24个cell:
    ```python
        from gensim.models import Word2Vec
        from gensim.models import KeyedVectors
        word2vec_model = 
        KeyedVectors.load_word2vec_format('''__embedding.txt文件存放的地址''')


    ```
    - 词向量是长度为100的向量。
    - 可以根据需要修改词向量的长度
