import mindspore
import word2vec
_one_hot = mindspore.ops.OneHot()
_ones = mindspore.ops.Ones()
_relu = mindspore.ops.ReLU()
class ResNet18(mindspore.nn.Cell):
    def _block(self, i, o):
        return mindspore.nn.SequentialCell([
            mindspore.nn.Conv2d(i, o, 3, o // i),
            mindspore.nn.ReLU(),
            mindspore.nn.Conv2d(o, o, 3),
        ])
    def __init__(self, size_image, size_feature):
        super().__init__()
        self.conv1 = mindspore.nn.SequentialCell([
            mindspore.nn.Conv2d(3, 64, 7, 2),
            mindspore.nn.MaxPool2d(3, 2, 'same'),
        ])
        self.conv2_1 = self._block(64, 64)
        self.conv2_2 = self._block(64, 64)
        self.resize3 = mindspore.nn.Conv2d(64, 128, 1, 2)
        self.conv3_1 = self._block(64, 128)
        self.conv3_2 = self._block(128, 128)
        self.resize4 = mindspore.nn.Conv2d(128, 256, 1, 2)
        self.conv4_1 = self._block(128, 256)
        self.conv4_2 = self._block(256, 256)
        self.resize5 = mindspore.nn.Conv2d(256, 512, 1, 2)
        self.conv5_1 = self._block(256, 512)
        self.conv5_2 = self._block(512, 512)
        size = -(-size_image // 32)
        self.end = mindspore.nn.SequentialCell([
            mindspore.nn.AvgPool2d(size),
            mindspore.nn.Flatten(),
            mindspore.nn.Dense(512, size_feature),
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
class LSTM512(mindspore.nn.Cell):
    def __init__(self, size_word, size_feature):
        super().__init__()
        self.lstm = mindspore.nn.LSTM(
            size_word, 512, 1,
            batch_first = True,
            bidirectional = True,
        )
        self.end = mindspore.nn.SequentialCell([
            mindspore.nn.Flatten(),
            mindspore.nn.Dense(8192, size_feature),
        ])
    def construct(self, x):
        ones = _ones((2, x.shape[0], 512), mindspore.float32)
        x = self.lstm(x, (ones, ones))[0]
        x = _relu(x)
        x = self.end(x)
        return x
class VQANet(mindspore.nn.Cell):
    def __init__(self, size_image = 224, size_word = 100, size_feature = 1024):
        '''e.g.
        vqa_net = VQANet(224, 8, 100, 1024)
        # result of the above example:
        # images are in shape 3, 224, 224
        # questions are in shape 8, 100
        # internal features are in shape 1024,
        '''
        super().__init__()
        self.size_image = size_image
        self.size_word = size_word
        self.size_feature = size_feature
        self.cell_image = ResNet18(size_image, size_feature)
        self.cell_question = LSTM512(size_word, size_feature)
        self.cell_feature = mindspore.nn.SequentialCell([
            mindspore.nn.Dense(size_feature, word2vec.size_vocabulary),
            mindspore.nn.Softmax(),
        ])
    def construct(self, image, question):
        image = self.cell_image(image)
        question = self.cell_question(question)
        feature = image * question
        feature = _relu(feature)
        feature = self.cell_feature(feature)
        return feature
    def accuracy(self, dataset):
        n_hits = 0
        n_rows = 0
        for image, question, answer in dataset:
            prediction = self.construct(image, question)
            n_hits += sum(prediction.asnumpy().argmax(1) == answer)
            n_rows += len(answer)
        return n_hits / n_rows
class VQALoss(mindspore.nn.Cell):
    def __init__(self, net):
        super().__init__(False)
        self.net = net
        self.loss = mindspore.nn.BCEWithLogitsLoss()
        self._0 = mindspore.Tensor(0, mindspore.float32)
        self._1 = mindspore.Tensor(1, mindspore.float32)
    def construct(self, image, question, answer):
        prediction = self.net(image, question)
        answer = _one_hot(answer, word2vec.size_vocabulary, self._1, self._0)
        return self.loss(prediction, answer)
