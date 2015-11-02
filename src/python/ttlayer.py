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
    def __init__(self, incoming, input_shape, output_shape, ranks, cores=lasagne.init.Normal(0.01), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(TTLayer, self).__init__(incoming, **kwargs)
        self.tt_input_shape = input_shape
        self.tt_output_shape = output_shape
        self.tt_ranks = ranks
        self.nonlinearity = nonlinearity
        self.num_dim = len(input_shape)
        cores_arr_len = np.sum(np.array(input_shape) * np.array(output_shape) * np.array(ranks[1:]) * np.array(ranks[:-1]))
        self.cores_arr = self.add_param(cores, (cores_arr_len, 1), name='cores_arr')

    def get_output_for(self, input, **kwargs):
        # theano.scan doesn't work when intermediate results' shape changes over
        # iterations (see https://github.com/Theano/Theano/issues/2127),
        # so we are using `for loop` instead.
        res = input
        core_idx = 0
        for k in range(self.num_dim):
            curr_shape = (self.tt_ranks[k] * self.tt_input_shape[k], self.tt_output_shape[k] * self.tt_ranks[k + 1])
            curr_core = self.cores_arr[core_idx:core_idx+T.prod(curr_shape)].reshape(curr_shape)
            res = T.dot(res.reshape((-1, curr_shape[0])), curr_core)
#             res = res.reshape((-1, self.output_shape[k])), transpose
            core_idx += T.prod(curr_shape)
        res = res.reshape((input.shape[0], -1))
        return self.nonlinearity(res)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], np.prod(self.tt_output_shape))
