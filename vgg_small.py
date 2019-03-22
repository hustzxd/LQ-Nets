import mxnet.gluon as mxg


class VGGSmallModel(mxg.HybridBlock):
    def __init__(self, classes=10, **kwargs):
        super(VGGSmallModel, self).__init__(**kwargs)

        # 32
        self.conv0 = mxg.nn.Conv2D(128, 3, padding=1, use_bias=False, prefix='conv0')
        self.bn1 = mxg.nn.BatchNorm(prefix='bn1')
        self.conv1 = mxg.nn.Conv2D(128, 3, padding=1, use_bias=False, prefix='conv1')

        self.bn2 = mxg.nn.BatchNorm(prefix='bn2')
        self.pool2 = mxg.nn.MaxPool2D(prefix='pool2')
        # 16
        self.conv2 = mxg.nn.Conv2D(256, 3, padding=1, use_bias=False, prefix='conv2')
        self.bn3 = mxg.nn.BatchNorm(prefix='bn3')
        self.conv3 = mxg.nn.Conv2D(256, 3, padding=1, use_bias=False, prefix='conv3')

        self.bn4 = mxg.nn.BatchNorm(prefix='bn4')
        self.pool4 = mxg.nn.MaxPool2D(prefix='pool4')
        # 8
        self.conv4 = mxg.nn.Conv2D(512, 3, padding=1, use_bias=False, prefix='conv4')
        self.bn5 = mxg.nn.BatchNorm(prefix='bn5')
        self.conv5 = mxg.nn.Conv2D(512, 3, padding=1, use_bias=False, prefix='conv5')

        self.bn6 = mxg.nn.BatchNorm(prefix='bn6')
        self.pool6 = mxg.nn.MaxPool2D(prefix='pool6')
        # 4

        self.fc = mxg.nn.Dense(classes, prefix='fc')

    def hybrid_forward(self, F, x, *args, **kwargs):
        # 32
        layer = self.conv0(x)
        layer = F.relu(self.bn1(layer))
        layer = self.conv1(layer)

        layer = F.relu(self.bn2(layer))
        layer = self.pool2(layer)
        # 16
        layer = self.conv2(layer)
        layer = F.relu(self.bn3(layer))
        layer = self.conv3(layer)

        layer = F.relu(self.bn4(layer))
        layer = self.pool4(layer)
        # 8
        layer = self.conv4(layer)
        layer = F.relu(self.bn5(layer))
        layer = self.conv5(layer)

        layer = F.relu(self.bn6(layer))
        layer = self.pool6(layer)
        # 4

        layer = self.fc(layer)
        return layer
