import mxnet.gluon as mxg
from learned_quantization import QuantizedActivation, Conv2DQ


class VGGSmallModelLq(mxg.HybridBlock):
    def __init__(self, classes=10, a_bits=2, w_bits=2, **kwargs):
        super(VGGSmallModelLq, self).__init__(**kwargs)
        # 32
        self.conv0 = mxg.nn.Conv2D(128, 3, padding=1, use_bias=False, prefix='conv0')
        self.bn1 = mxg.nn.BatchNorm(prefix='bn1')

        self.convq1 = Conv2DQ(nbits=w_bits, channels=128, kernel_size=3, in_channels=128, padding=1, use_bias=False,
                              prefix='convq1')
        # self.conv1 = mxg.nn.Conv2D(128, 3, padding=1, use_bias=False, in_channels=128, prefix='conv1')

        self.bn2 = mxg.nn.BatchNorm(prefix='bn2')
        self.pool2 = mxg.nn.MaxPool2D(prefix='pool2')
        # 16
        # self.conv2 = mxg.nn.Conv2D(256, 3, padding=1, use_bias=False, prefix='conv2')
        self.convq2 = Conv2DQ(nbits=w_bits, channels=256, kernel_size=3, in_channels=128, padding=1, use_bias=False,
                              prefix='convq2')
        self.bn3 = mxg.nn.BatchNorm(prefix='bn3')
        # self.conv3 = mxg.nn.Conv2D(256, 3, padding=1, use_bias=False, prefix='conv3')
        self.convq3 = Conv2DQ(nbits=w_bits, channels=256, kernel_size=3, in_channels=256, padding=1, use_bias=False,
                              prefix='convq3')
        self.bn4 = mxg.nn.BatchNorm(prefix='bn4')
        self.pool4 = mxg.nn.MaxPool2D(prefix='pool4')
        # 8
        # self.conv4 = mxg.nn.Conv2D(512, 3, padding=1, use_bias=False, prefix='conv4')
        self.convq4 = Conv2DQ(nbits=w_bits, channels=512, kernel_size=3, in_channels=256, padding=1, use_bias=False,
                              prefix='convq4')
        self.bn5 = mxg.nn.BatchNorm(prefix='bn5')
        # self.conv5 = mxg.nn.Conv2D(512, 3, padding=1, use_bias=False, prefix='conv5')
        self.convq5 = Conv2DQ(nbits=w_bits, channels=512, kernel_size=3, in_channels=512, padding=1, use_bias=False,
                              prefix='convq5')
        self.bn6 = mxg.nn.BatchNorm(prefix='bn6')
        self.pool6 = mxg.nn.MaxPool2D(prefix='pool6')
        # 4

        self.fc = mxg.nn.Dense(classes, prefix='fc')

        self.actq1 = QuantizedActivation(nbits=a_bits, prefix='actq1')
        self.actq2 = QuantizedActivation(nbits=a_bits, prefix='actq2')
        self.actq3 = QuantizedActivation(nbits=a_bits, prefix='actq3')
        self.actq4 = QuantizedActivation(nbits=a_bits, prefix='actq4')
        self.actq5 = QuantizedActivation(nbits=a_bits, prefix='actq5')

    def hybrid_forward(self, F, x, *args, **kwargs):
        # 32
        layer = self.conv0(x)
        layer = F.relu(self.bn1(layer))

        layer = self.actq1(layer)
        layer = self.convq1(layer)
        # 16
        layer = F.relu(self.bn2(layer))
        layer = self.pool2(layer)

        layer = self.actq2(layer)
        layer = self.convq2(layer)
        layer = F.relu(self.bn3(layer))
        layer = self.actq3(layer)
        layer = self.convq3(layer)

        layer = F.relu(self.bn4(layer))
        layer = self.pool4(layer)
        # 8
        layer = self.actq4(layer)
        layer = self.convq4(layer)
        layer = F.relu(self.bn5(layer))

        layer = self.actq5(layer)
        layer = self.convq5(layer)

        layer = F.relu(self.bn6(layer))
        layer = self.pool6(layer)
        # 4

        layer = self.fc(layer)
        return layer
