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
    def __init__(self, incoming, tt_input_shape, tt_output_shape, tt_ranks,
                 cores=lasagne.init.Normal(0.01),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(TTLayer, self).__init__(incoming, **kwargs)
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
        cores_arr_len = np.sum(tt_input_shape * tt_output_shape * tt_ranks[1:] * tt_ranks[:-1])
        self.cores_arr = self.add_param(cores, (cores_arr_len, 1), name='cores_arr')

    def get_output_for(self, input, **kwargs):
        # theano.scan doesn't work when intermediate results' shape changes over
        # iterations (see https://github.com/Theano/Theano/issues/2127),
        # so we are using `for loop` instead.
        res = input
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
        res = res.reshape((input.shape[0], -1))
        # res is of size batch_size x o_1 x ... x o_d
        return self.nonlinearity(res)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(self.tt_output_shape))
