import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
import theano

class SimpleSampleLayer(lasagne.layers.MergeLayer):
    """
    Simple sampling layer drawing a single Monte Carlo sample to approximate
    E_q [log( p(x,z) / q(z|x) )]. This is the approach described in [KINGMA]_.
    Parameters
    ----------
    mu, log_var : :class:`Layer` instances
        Parameterizing the mean and log(variance) of the distribution to sample
        from as described in [KINGMA]_. The code assumes that these have the
        same number of dimensions.
    seed : int
        seed to random stream
    Methods
    ----------
    seed : Helper function to change the random seed after init is called
    References
    ----------
        ..  [KINGMA] Kingma, Diederik P., and Max Welling.
            "Auto-Encoding Variational Bayes."
            arXiv preprint arXiv:1312.6114 (2013).
    """
    def __init__(self, mean, log_var,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(SimpleSampleLayer, self).__init__([mean, log_var], **kwargs)

        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
       self._srng.seed(seed)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        mu, log_var = input
        eps = self._srng.normal(mu.shape)
        if deterministic:
            eps *= 0.01
        z = mu + T.exp(0.5 * log_var) * eps
        return z
