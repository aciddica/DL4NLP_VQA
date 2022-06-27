'''
!
TODO 2
!
fill in the following functions
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
