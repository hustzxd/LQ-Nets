import mxnet as mx
import mxnet.gluon as mxg
from mxnet import autograd
from mxnet.gluon import HybridBlock
from numpy.linalg import inv
import torch

MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745


class MatrixInverse(mx.operator.CustomOp):
    """
    Note: Unluckily, I did not find the inverse of the matrix operation in MXNet.
    In the end, I add custom operation to implement matrix inverse.
    Reference: https://mxnet.incubator.apache.org/tutorials/gluon/customop.html
    """

    def forward(self, is_train, req, in_data, out_data, aux):
        """Implements forward computation.

        is_train : bool, whether forwarding for training or testing.
        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to out_data. 'null' means skip assignment, etc.
        in_data : list of NDArray, input data.
        out_data : list of NDArray, pre-allocated output buffers.
        aux : list of NDArray, mutable auxiliary states. Usually not used.
        """
        x = in_data[0].asnumpy()
        # TODO: need cuda????? cost much time
        # TODO: if k == 1, B_inv = 1/B
        # TODO: if k == 2, B_inv = 1/(ad-bc) * [[d, -b],[-c, a]]
        if x.shape[0] == 1:
            try:
                x_inv = inv(x)
            except:
                print('activation: {} is a singular matrix \n try + 1'.format(x))
                x[0, 0, 0] += 1
                try:
                    x_inv = inv(x)
                except:
                    print('!!!activation: {} is a singular matrix \n exit now!!'.format(x))
                    exit()
            x_inv = mx.nd.array(x_inv)
        else:
            torch_x = torch.Tensor(x)
            torch_y = torch.zeros_like(torch_x)
            for i in range(torch_x.shape[0]):
                try:
                    torch_y[i] = torch_x[i].inverse()
                except:
                    print('weights: {} is a singular matrix \n try + 1'.format(torch_y[i]))
                    torch_x[i, 0, 0] += 1
                    try:
                        torch_y[i] = torch_x[i].inverse()
                    except:
                        print('!!!weights: {} is a singular matrix \n exit now!!'.format(torch_x[i]))
                        exit()
            x_inv = torch_y.numpy()
            x_inv = mx.nd.array(x_inv)
        self.assign(out_data[0], req[0], x_inv)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        """Implements backward computation

        req : list of {'null', 'write', 'inplace', 'add'}, how to assign to in_grad
        out_grad : list of NDArray, gradient w.r.t. output data.
        in_grad : list of NDArray, gradient w.r.t. input data. This is the output buffer.
        """
        dy = out_grad[0].asnumpy()
        # Note: We don't need backward in this projects.
        self.assign(in_grad[0], req[0], mx.nd.array(dy))


@mx.operator.register("matrix_inverse")  # register with name "sigmoid"
class MatrixInverseProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MatrixInverseProp, self).__init__(True)

    def list_arguments(self):
        #  this can be omitted if you only have 1 input.
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        """Calculate output shapes from input shapes. This can be
        omited if all your inputs and outputs have the same shape.

        in_shapes : list of shape. Shape is described by a tuple of int.
        """
        data_shape = in_shapes[0]
        assert len(data_shape) == 3, 'matrix inverse only support 2-dim matrix with n channel'
        assert data_shape[1] == data_shape[2], 'matrix inverse only support matrix that has same col and row'
        output_shape = data_shape
        # return 3 lists representing inputs shapes, outputs shapes, and aux data shapes.
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return MatrixInverse()


class QuantizedActivation(HybridBlock):
    def __init__(self, nbits, op_name='ActQ', prefix=None, params=None):
        super(QuantizedActivation, self).__init__(prefix=prefix, params=params)
        if nbits <= 0:
            return
        self.nbits = nbits
        init_basis = [[(NORM_PPF_0_75 * 2 / (2 ** self.nbits - 1)) * (2. ** i)] for i in range(self.nbits)]
        init_basis = mx.init.Constant(init_basis)
        bit_dims = [self.nbits, 1]
        self.num_levels = 2 ** self.nbits
        # initialize level multiplier
        init_level_multiplier = []
        for i in range(0, self.num_levels):
            level_multiplier_i = [0. for j in range(self.nbits)]
            level_number = i
            for j in range(self.nbits):
                level_multiplier_i[j] = float(level_number % 2)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        init_level_multiplier = mx.init.Constant(init_level_multiplier)
        # initialize threshold multiplier
        init_thrs_multiplier = []
        for i in range(1, self.num_levels):
            thrs_multiplier_i = [0. for j in range(self.num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)
        init_thrs_multiplier = mx.init.Constant(init_thrs_multiplier)
        with self.name_scope():
            self.basis = self.params.get('basis',
                                         shape=bit_dims,
                                         init=init_basis,
                                         # Convert to regular list to make this object serializable
                                         allow_deferred_init=False,
                                         differentiable=False)
            self.level_codes = self.params.get('level_codes',
                                               shape=[2 ** nbits, nbits],
                                               init=init_level_multiplier,
                                               allow_deferred_init=False,
                                               differentiable=False)
            self.thrs_multiplier = self.params.get('thrs_multiplier',
                                                   shape=[2 ** nbits - 1, 2 ** nbits],
                                                   init=init_thrs_multiplier,
                                                   allow_deferred_init=False,
                                                   differentiable=False)

    def hybrid_forward(self, F, x, basis=None, level_codes=None, thrs_multiplier=None):
        # print('basis:{}'.format(basis))
        if basis is None:
            return x
        # calculate levels and sort
        with autograd.pause():
            levels = F.dot(level_codes, basis)
            levels, sort_id = F.topk(F.transpose(levels), ret_typ='both', k=self.num_levels, is_ascend=1)  # ascend
            levels = F.transpose(levels)
            # TODO: levels need backward
            sort_id = F.transpose(sort_id)
            # calculate threshold
            thrs = F.dot(thrs_multiplier, levels)
            # calculate output y and its binary code
            y = F.zeros_like(x)  # output
            reshape_x = F.reshape(x, [-1])
            BT = F.zeros_like(reshape_x)
            BT = F.reshape(F.repeat(BT, self.nbits), shape=(-1, self.nbits))  # (N, k)
            zero_y = F.zeros_like(x)
            zero_bits_y = F.zeros_like(BT)
            for i in range(self.num_levels - 1):
                g = F.broadcast_greater(x, thrs[i])  # module 'mxnet.symbol' has no attribute 'greater'
                y = F.where(g, zero_y + levels[i + 1], y)
                BT = F.where(F.reshape(g, [-1]), zero_bits_y + level_codes[sort_id[i + 1][0]], BT)
        if autograd.is_training():
            with autograd.pause():
                # calculate BxBT
                B = F.transpose(BT)
                BxBT = F.zeros([self.nbits, self.nbits])
                for i in range(self.nbits):
                    for j in range(self.nbits):
                        BxBTij = F.multiply(B[i], B[j])
                        BxBTij = F.sum(BxBTij)
                        if i == j:
                            BxBTij += EPS
                        BxBT[i, j] = BxBTij
                BxBT_inv = F.Custom(BxBT.expand_dims(0), op_type='matrix_inverse')
                BxBT_inv = BxBT_inv[0]
                # BxBT_inv = BxBT
                # calculate BxX
                BxX = F.zeros([self.nbits])
                for i in range(self.nbits):
                    BxXi0 = F.multiply(B[i], reshape_x)
                    BxXi0 = F.sum(BxXi0)
                    BxX[i] = BxXi0
                BxX = F.reshape(BxX, [self.nbits, 1])
                new_basis = F.dot(BxBT_inv, BxX)
                # create moving averages op
                basis = MOVING_AVERAGES_FACTOR * basis + new_basis * (1 - MOVING_AVERAGES_FACTOR)
                self.basis.set_data(basis)
        x_clip = F.minimum(x, levels[self.num_levels - 1])  # gradient clip
        y = x_clip + F.stop_gradient(-x_clip) + F.stop_gradient(y)  # gradient: y=clip(x)
        return y


class Conv2DQ(mxg.nn.Conv2D):
    def __init__(self, nbits, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        super(Conv2DQ, self).__init__(channels, kernel_size, strides=strides, padding=padding,
                                      dilation=dilation, groups=groups, layout=layout,
                                      activation=activation, use_bias=use_bias, weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer, in_channels=in_channels, **kwargs)
        if nbits <= 0:
            return
        self._nbits = nbits
        self._num_filters = channels
        base = NORM_PPF_0_75 * ((2. / (channels * kernel_size * kernel_size)) ** 0.5) / (2 ** (nbits - 1))
        init_basis = []
        for j in range(nbits):
            init_basis.append([(2 ** j) * base for i in range(channels)])
        init_basis = mx.init.Constant(init_basis)
        # initialize level multiplier
        self._num_levels = 2 ** nbits
        init_level_multiplier = []
        for i in range(self._num_levels):
            level_multiplier_i = [0. for j in range(nbits)]
            level_number = i
            for j in range(nbits):
                binary_code = level_number % 2
                if binary_code == 0:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        init_level_multiplier = mx.init.Constant(init_level_multiplier)
        # initialize threshold multiplier
        init_thrs_multiplier = []
        for i in range(1, self._num_levels):
            thrs_multiplier_i = [0. for j in range(self._num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)
        init_thrs_multiplier = mx.init.Constant(init_thrs_multiplier)
        init_sum_multiplier = mx.nd.ones(shape=(1, kernel_size * kernel_size * in_channels))
        init_sum_multiplier = mx.init.Constant(init_sum_multiplier)
        init_sum_multiplier_basis = mx.nd.ones(shape=[1, nbits])
        init_sum_multiplier_basis = mx.init.Constant(init_sum_multiplier_basis)
        with self.name_scope():
            self.basis = self.params.get('basis',
                                         shape=(nbits, channels),
                                         init=init_basis,
                                         allow_deferred_init=False,
                                         differentiable=False)
            self.level_code = self.params.get('level_code',
                                              shape=(self._num_levels, nbits),
                                              init=init_level_multiplier,
                                              allow_deferred_init=False,
                                              differentiable=False)
            self.thrs_multiplier = self.params.get('thrs_multiplier',
                                                   shape=(self._num_levels - 1, self._num_levels),
                                                   init=init_thrs_multiplier,
                                                   allow_deferred_init=False,
                                                   differentiable=False)
            self.sum_multiplier = self.params.get('sum_multiplier',
                                                  shape=(1, kernel_size * kernel_size * in_channels),
                                                  init=init_sum_multiplier,
                                                  allow_deferred_init=False,
                                                  differentiable=False)
            self.sum_multiplier_basis = self.params.get('sum_multiplier_basis',
                                                        shape=(1, nbits),
                                                        init=init_sum_multiplier_basis,
                                                        allow_deferred_init=False,
                                                        differentiable=False)

    def hybrid_forward(self, F, x, weight, bias=None, basis=None, level_code=None, thrs_multiplier=None,
                       sum_multiplier=None, sum_multiplier_basis=None):
        if basis is None:
            return super(Conv2DQ, self).hybrid_forward(F, x, weight=weight, bias=bias)
        with autograd.pause():
            tf_w = F.transpose(weight, [2, 3, 1, 0])  # transpose w as [h, w, in_c, out_c]
            levels = F.dot(level_code, basis)
            levels, sort_id = F.topk(F.transpose(levels), ret_typ='both', k=self._num_levels, is_ascend=1)  # ascend
            levels = F.transpose(levels)
            sort_id = F.transpose(sort_id)
            # calculate threshold
            thrs = F.dot(thrs_multiplier, levels)

            level_codes_channelwise = F.zeros(shape=(self._num_levels * self._num_filters, self._nbits))
            for i in range(self._num_levels):
                eq = F.equal(sort_id, i)  # (4, 128)
                # tf.reshape(eq, [-1]) (512, )
                level_codes_channelwise = F.where(F.reshape(eq, [-1]), level_codes_channelwise + level_code[i],
                                                  level_codes_channelwise)
            level_codes_channelwise = F.reshape(level_codes_channelwise,
                                                [self._num_levels, self._num_filters, self._nbits])
            reshape_w = F.reshape(tf_w, [-1, self._num_filters])  # (576, 128)
            w_q = F.zeros_like(tf_w) + levels[0]  # 3x3x64x128
            w_q_zero = F.zeros_like(tf_w)  # 3x3x64x128
            zero_dims = [reshape_w.shape[0] * self._num_filters, self._nbits]  # [576 * 128, 2]
            bits_w_q = F.zeros(zero_dims) + -1  # (576 * 128, 2)
            zero_bits_w_q = F.reshape(F.zeros(zero_dims), [-1, self._num_filters, self._nbits])  # (576, 128, 2)
            for i in range(self._num_levels - 1):
                g = F.greater(tf_w, thrs[i])
                w_q = F.where(g, w_q_zero + levels[i + 1], w_q)
                bits_w_q = F.where(F.reshape(g, [-1]),
                                   # zero_bits_w_q + level_codes_channelwise[i + 1] (576,128,2)
                                   F.reshape(zero_bits_w_q + level_codes_channelwise[i + 1], [-1, self._nbits]),
                                   bits_w_q)
            bits_w_q = F.reshape(bits_w_q, [-1, self._num_filters, self._nbits])  # (576, 128, 2)
        if autograd.is_training():
            with autograd.pause():
                # sum_multiplier = F.ones([1, F.reshape(tf_w, [-1, self._num_filters]).shape[0]])
                # sum_multiplier_basis = F.ones([1, self._nbits])
                BT = F.transpose(bits_w_q, [2, 0, 1])
                # calculate BTxB
                BTxB = F.zeros([self._nbits, self._nbits, self._num_filters])
                for i in range(self._nbits):
                    for j in range(self._nbits):
                        # TODO: BTxBij == BTxBji, so it can speed up
                        BTxBij = F.multiply(BT[i], BT[j])
                        BTxBij = F.dot(sum_multiplier, BTxBij)
                        if i == j:
                            mat_one = F.ones([1, self._num_filters])
                            BTxBij = BTxBij + (EPS * mat_one)  # + E
                        BTxB[i, j, :] = F.reshape(BTxBij, [-1])
                # calculate inverse of BTxB
                BTxB_inv = F.zeros([self._nbits, self._nbits, self._num_filters])
                if self._nbits > 2:
                    BTxB_transpose = F.transpose(BTxB, [2, 0, 1])
                    BTxB_inv = F.Custom(BTxB_transpose, op_type='matrix_inverse')
                    BTxB_inv = F.transpose(BTxB_inv, [1, 2, 0])
                elif self._nbits == 2:
                    det = F.multiply(BTxB[0][0], BTxB[1][1]) - F.multiply(BTxB[0][1], BTxB[1][0])
                    BTxB_inv[0, 0] = BTxB[1][1] / det
                    BTxB_inv[0, 1] = -BTxB[0][1] / det
                    BTxB_inv[1, 0] = -BTxB[1][0] / det
                    BTxB_inv[1, 1] = BTxB[0][0] / det
                elif self._nbits == 1:
                    BTxB_inv = 1 / BTxB
                # calculate BTxX
                BTxX = F.zeros([self._nbits, self._num_filters])
                for i in range(self._nbits):
                    BTxXi0 = F.multiply(BT[i], reshape_w)  # (576, 128)
                    BTxXi0 = F.dot(sum_multiplier, BTxXi0)
                    BTxX[i] = BTxXi0
                new_basis = F.zeros([self._nbits, self._num_filters])

                for i in range(self._nbits):
                    new_basis_i = F.multiply(BTxB_inv[i], BTxX)
                    new_basis_i = F.dot(sum_multiplier_basis, new_basis_i)
                    new_basis[i] = new_basis_i
                # create moving averages op
                basis = MOVING_AVERAGES_FACTOR * basis + new_basis * (1 - MOVING_AVERAGES_FACTOR)
                self.basis.set_data(basis)
        weight_q = F.transpose(w_q, [3, 2, 0, 1])
        weight_q = weight + F.stop_gradient(-weight) + F.stop_gradient(weight_q)  # gradient: y=x
        return super(Conv2DQ, self).hybrid_forward(F, x, weight=weight_q, bias=bias)


def print_params(title, net):
    """
    Helper function to print out the state of parameters.
    """
    print(title)
    for key, value in net.collect_params().items():
        data = value.data().reshape(-1)
        if len(data) > 100:
            print('{} = {} shape:{}\n'.format(key, data[0:100], value.data().shape))
        else:
            print('{} = {}\n'.format(key, value.data()))
