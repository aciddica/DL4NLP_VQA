
import numpy
import mindspore
class VQANet(mindspore.nn.Cell):
    def __init__(self, size_image, size_word, size_question, size_answer, size_feature):
        '''e.g.
        vqa_net = VQANet(256, 100, 8, 4, 4096)
        # result of the above example:
        # images are in shape 3, 128, 128
        # questions are in shape 8, 100
        # answers are in shape 4, 100
        # internal features are in shape 4096,
        '''
        super().__init__()
        self.size_image = size_image
        self.size_word = size_word
        self.size_question = size_question
        self.size_answer = size_answer
        self.size_feature = size_feature
        self.cell_image = mindspore.nn.SequentialCell([
            mindspore.nn.Conv2d(3, ...),
            mindspore.nn.ReLU(),
            ...,
            mindspore.nn.Flatten(),
            mindspore.nn.Dense(...),
            mindspore.nn.ReLU(),
            ...,
            mindspore.nn.Dense(..., size_feature),
        ])
        self.cell_question = mindspore.nn.SequentialCell([
            ...,
            mindspore.nn.Dense(..., size_feature),
        ])
        self.cell_feature = mindspore.nn.SequentialCell([
            mindspore.nn.ReLU(),
            mindspore.nn.Dense(size_feature, ...),
            mindspore.nn.ReLU(),
            mindspore.nn.Dense(..., size_word * size_answer),
        ])
    def construct(self, image, question):
        image = self.cell_image(image)
        question = self.cell_question(question)
        feature = image * question # tbd, some attention operation
        feature = self.cell_feature(feature)
        return feature.reshape((feature.shape[0], self.size_answer, self.size_word))
    def answer(self, image, question):
        answer = self.construct(image, question)
        answer = ...(answer)
        # decode mindspore.Tensor in shape (size_batch, size_answer, size_word) to tuple of str with length size_batch
        return answer
class Loss(mindspore.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def construct(self, image, question, answer):
        prediction = self.net(image, question)
        loss = ... # some function to compare prediction and answer
        return loss
vqa_net = VQANet(...)
loss = Loss(vqa_net)
optimizer = mindspore.nn.SGD(loss.trainable_params()) # or some customized optimizer
model = mindspore.Model(loss, None, optimizer)
model.train(n_iterations, dataset_train, ...)
