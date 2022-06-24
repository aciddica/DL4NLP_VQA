
 **在notebook环境下运行无误，但是换过文件夹环境，可能会出现路径不对、变量未定义等问题**
 **load_json_file函数测试过没有问题**

 **目前是用dataframe存储的questi id，image id，question和它的张量，annotation和它的张量**
 - utils.py
    一些统计问题id等的功能函数

    如果需要处理数据集删除没有对应图片的问题，直接调用utils.gen_clean_dataset()即可
    以及load_json_file可能有用

    其他函数不怎么用得上
    
- work_space.ipynb
    打的草稿
    以及一些config

- word2vec.py
    词向量模型的产生和load

