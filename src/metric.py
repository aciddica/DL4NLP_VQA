def decode(tensor):
    '''
    !
    TODO 3
    !
    tensor: mindspore.Tensor in shape (size_batch, length_output_vector)
    this is the raw output of a VQANet
    decode each row to a str, and return a tuple of strs
    e.g.
    tensor = mindspore.Tensor([[0.2, 2.5, ...], [-3.1, 0, ...]])
    decode(tensor) -> 'yes', 'two'
    '''
def accuracy(tensor, answer):
    '''
    !
    TODO 4
    !
    tensor: mindspore.Tensor in shape (size_batch, length_output_vector)
    this is the raw output of a VQANet
    answer: mindspore.Tensor in shape ((size_batch,) + shape_answer)
    this is the annotation provided by a VQASet
    returns the proportion of correct predictions
    e.g.
    tensor = mindspore.Tensor([
        [0.2, 2.5, ...], # correct
        [-3.1, 0, ...], # wrong
    ])
    answer = mindspore.Tensor([annotation1, annotation2])
    accuracy(tensor, answer) -> 0.5
    '''
