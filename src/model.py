import mindspore
_relu = mindspore.ops.ReLU()
class ResNet18(mindspore.nn.Cell):
    def _block(n):
        return mindspore.nn.SequentialCell([
            mindspore.nn.Conv2d(n, n, 3),
            mindspore.nn.ReLU(),
            mindspore.nn.Conv2d(n, n, 3),
        ])
    def __init__(self, size_image, size_feature):
        super().__init__()
        self.conv1 = mindspore.nn.SequentialCell([
            mindspore.nn.Conv2d(3, 64, 7, 2),
            mindspore.nn.MaxPool2d(3, 2, 'same'),
        ])
        self.conv2_1 = self._block(64)
        self.conv2_2 = self._block(64)
        self.resize3 = mindspore.nn.Conv2d(64, 128, 1, 2)
        self.conv3_1 = self._block(128)
        self.conv3_2 = self._block(128)
        self.resize4 = mindspore.nn.Conv2d(128, 256, 1, 2)
        self.conv4_1 = self._block(256)
        self.conv4_2 = self._block(256)
        self.resize5 = mindspore.nn.Conv2d(256, 512, 1, 2)
        self.conv5_1 = self._block(512)
        self.conv5_2 = self._block(512)
        size = -(-size_image // 32)
        self.end = mindspore.nn.SequentialCell([
            mindspore.nn.AvgPool2d(size),
            mindspore.nn.Flatten(),
            mindspore.nn.Dense(size, size_feature),
        ])
    def construct(self, x):
        x = self.conv1(x)
        x = _relu(x)
        x += self.conv2_1(x)
        x = _relu(x)
        x += self.conv2_2(x)
        x = _relu(x)
        x = self.resize3(x) + self.conv3_1(x)
        x = _relu(x)
        x += self.conv3_2(x)
        x = _relu(x)
        x = self.resize4(x) + self.conv4_1(x)
        x = _relu(x)
        x += self.conv4_2(x)
        x = _relu(x)
        x = self.resize5(x) + self.conv5_1(x)
        x = _relu(x)
        x += self.conv5_2(x)
        x = _relu(x)
        x = self.end(x)
        return x
class VQANet(mindspore.nn.Cell):
    def __init__(self, size_image, size_question, size_word, size_feature, size_output):
        '''e.g.
        vqa_net = VQANet(224, 8, 100, 1024, 1024)
        # result of the above example:
        # images are in shape 3, 224, 224
        # questions are in shape 8, 100
        # internal features are in shape 1024,
        # output vectors are in shape 1024,
        '''
        super().__init__()
        self.size_image = size_image
        self.size_question = size_question
        self.size_word = size_word
        self.size_feature = size_feature
        self.size_output = size_output
        self.cell_image = ResNet18(size_image, size_feature)
        self.cell_question = mindspore.nn.SequentialCell([
            ...,
            mindspore.nn.Dense(..., size_feature),
        ])
        self.cell_feature = mindspore.nn.SequentialCell([
            mindspore.nn.Dense(size_feature, size_output),
            mindspore.nn.Softmax(),
        ])
    def construct(self, image, question):
        image = self.cell_image(image)
        question = self.cell_question(question)
        feature = image * question # tbd, some attention operation
        feature = _relu(feature)
        feature = self.cell_feature(feature)
        return feature
    def answer(self, image, question):
        answer = self.construct(image, question)
        answer = ...(answer)
        # decode mindspore.Tensor in shape (size_batch, size_output) to size_batch-tuple of str
        return answer
class Loss(mindspore.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def construct(self, image, question, answer):
        prediction = self.net(image, question)
        loss = ... # some function to compare prediction and answer
        return loss
vqa_net = VQANet(224, 8, 100, 1024, 1024)
loss = Loss(vqa_net)
optimizer = mindspore.nn.SGD(loss.trainable_params()) # or some customized optimizer
model = mindspore.Model(loss, None, optimizer)
model.train(n_iterations, dataset_train, ...)
