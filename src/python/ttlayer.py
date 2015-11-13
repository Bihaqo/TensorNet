import numpy as np
import theano
import theano.tensor as T
import lasagne

class TTLayer(lasagne.layers.Layer):
    """
    Parameters
    ----------
    References
    ----------
    .. [1]  Tensorizing Neural Networks
        Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov
        In Advances in Neural Information Processing Systems 28 (NIPS-2015)
    Notes
    -----
    Examples
    --------
    """
    # TODO: add biases.
    def __init__(self, incoming, tt_input_shape, tt_output_shape, tt_ranks,
                 cores=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(TTLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        num_inputs = int(np.prod(self.input_shape[1:]))
        tt_input_shape = np.array(tt_input_shape)
        tt_output_shape = np.array(tt_output_shape)
        tt_ranks = np.array(tt_ranks)
        if np.prod(tt_input_shape) != num_inputs:
            raise ValueError("The size of the input tensor (i.e. product "
                             "of the elements in tt_input_shape) should "
                             "equal to the number of input neurons %d." %
                             (num_inputs))
        if tt_input_shape.shape[0] != tt_output_shape.shape[0]:
            raise ValueError("The number of input and output dimensions "
                             "should be the same.")
        if tt_ranks.shape[0] != tt_output_shape.shape[0] + 1:
            raise ValueError("The number of the TT-ranks should be "
                             "1 + the number of the dimensions.")
        self.tt_input_shape = tt_input_shape
        self.tt_output_shape = tt_output_shape
        self.tt_ranks = tt_ranks
        self.nonlinearity = nonlinearity
        self.num_dim = tt_input_shape.shape[0]
        local_cores_arr = _generate_orthogonal_tt_cores(tt_input_shape,
                                                       tt_output_shape,
                                                       tt_ranks)
        self.cores_arr = self.add_param(local_cores_arr, local_cores_arr.shape, name='cores_arr')

    def get_output_for(self, input, **kwargs):
        # theano.scan doesn't work when intermediate results' shape changes over
        # iterations (see https://github.com/Theano/Theano/issues/2127),
        # so we are using `for loop` instead.
        res = input
        # TODO: it maybe faster to precompute the indices in advance.
        core_arr_idx = 0
        for k in range(self.num_dim - 1, -1, -1):
            # res is of size o_k+1 x ... x o_d x batch_size x i_1 x ... x i_k-1 x i_k x r_k+1
            curr_shape = (self.tt_input_shape[k] * self.tt_ranks[k + 1], self.tt_ranks[k] * self.tt_output_shape[k])
            curr_core = self.cores_arr[core_arr_idx:core_arr_idx+T.prod(curr_shape)].reshape(curr_shape)
            res = T.dot(res.reshape((-1, curr_shape[0])), curr_core)
            # res is of size o_k+1 x ... x o_d x batch_size x i_1 x ... x i_k-1 x r_k x o_k
            res = T.transpose(res.reshape((-1, self.tt_output_shape[k])))
            # res is of size o_k x o_k+1 x ... x o_d x batch_size x i_1 x ... x i_k-1 x r_k
            core_arr_idx += T.prod(curr_shape)
        # res is of size o_1 x ... x o_d x batch_size
        res = T.transpose(res.reshape((-1, input.shape[0])))
        # res is of size batch_size x o_1 x ... x o_d
        return self.nonlinearity(res)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(self.tt_output_shape))


def _generate_orthogonal_tt_cores(input_shape, output_shape, ranks):
    # Generate random orthogonalized tt-tensor.
    input_shape = np.array(input_shape)
    output_shape = np.array(output_shape)
    ranks = np.array(ranks)
    cores_arr_len = np.sum(input_shape * output_shape *
                           ranks[1:] * ranks[:-1])
    cores_arr = lasagne.utils.floatX(np.zeros(cores_arr_len))
    cores_arr_idx = 0
    core_list = []
    rv = 1
    for k in range(input_shape.shape[0]):
        shape = [ranks[k], input_shape[k], output_shape[k], ranks[k+1]]
        tall_shape = (np.prod(shape[:3]), shape[3])
        curr_core = np.dot(rv, lasagne.random.get_rng().normal(0, 1, size=(shape[0], np.prod(shape[1:]))))
        curr_core = curr_core.reshape(tall_shape)
        if k < input_shape.shape[0]-1:
            curr_core, rv = np.linalg.qr(curr_core)
        cores_arr[cores_arr_idx:cores_arr_idx+curr_core.size] = curr_core.flatten()
        cores_arr_idx += curr_core.size
    # TODO: use something reasonable instead of this dirty hack.
    glarot_style = (np.prod(input_shape) * np.prod(ranks))**(1.0 / input_shape.shape[0])
    return (0.1 / glarot_style) * lasagne.utils.floatX(cores_arr)
