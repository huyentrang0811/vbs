import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib
from libs.util import ind_Bernoulli_fn, ind_multivariate_normal_fn, default_Bernoulli_fn
from libs.util import GumbelSoftMax_fn, KLDivergenceOfCorrespondingBernoulli

class Dropout_1(tf.keras.layers.Layer):
    def __init__(self,
                s_t,
                mask_posterior_fn,
                mask_prior_fn,
                mask_posterior_tensor_fn=lambda d: d.sample(),
                mask_divergence_fn=KLDivergenceOfCorrespondingBernoulli):
        super().__init__()
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.mask_posterior_fn = mask_posterior_fn
        self.mask_posterior_tensor_fn = mask_posterior_tensor_fn
        self.mask_prior_fn = mask_prior_fn
        self.mask_divergence_fn = mask_divergence_fn
        self.s_t = s_t

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        in_size = tf.compat.dimension_value(input_shape.with_rank_at_least(2)[-1])
        if in_size is None:
            raise ValueError('The last dimension of the inputs to `Dropout` '
                            'should be defined. Found `None`.')
        self._input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: in_size})

            # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        
            # Must have a posterior kernel.
        self.mask_posterior = self.mask_posterior_fn(
                dtype, [in_size], 'mask_posterior',
                self.trainable, self.add_variable)
        if self.mask_prior_fn is None:
            self.mask_prior = None
        else:
            self.mask_prior = self.mask_prior_fn(
                dtype, [in_size], 'mask_prior',
                self.trainable, self.add_variable)

        self.built = True

    def call(self, inputs, training=True):
        outputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)

        if self.s_t == 1:
            outputs = self._apply_variational_dropout(outputs, training)
            self._apply_divergence(
                self.mask_divergence_fn,
                self.mask_posterior,
                self.mask_prior,
                self.mask_posterior_tensor,
                training,
                name='divergence_mask')
        return outputs

    def _apply_variational_dropout(self, inputs, training):
        if training:
            self.mask_posterior_tensor = self.mask_posterior_tensor_fn(self.mask_posterior)
            #mask_posterior_tensor_shape = self.mask_posterior_tensor.get_shape().as_list()
            #print("mask_shape", mask_posterior_tensor_shape)
            return tf.math.multiply(inputs, self.mask_posterior_tensor[:, -1])

        else:
            self.mask_posterior_tensor = self.mask_posterior_tensor_fn(self.mask_posterior) #avoid bug
            #print("******")
            #print(tf.keras.backend.get_value(self.mask_posterior.distribution.probs))
            in_size = inputs.get_shape().as_list()[1]
            return inputs * (tf.ones(shape = in_size) - self.mask_posterior.distribution.probs[:, 0])


    def _apply_divergence(self, divergence_fn, posterior, prior,
                            posterior_tensor, training, name, gamma = 0.8):
        if (divergence_fn is not None and
            posterior is not None and
            prior is not None and
            self.s_t==1 and 
            training):
            divergence = tf.identity(
                divergence_fn(
                posterior, prior, posterior_tensor),
                name=name)
            self.add_loss(gamma * divergence)
if __name__=="__main__":
    initial_prior_droprate = 0.5
    prior_dropoutrate = [0.4]

    initial_prior_var = 1.0
    prior_m = [None]
    prior_s = [None]
    
    kwargs = {
        'kernel_posterior_tensor_fn': (
            lambda d: d.sample()),
    }

    f = Dropout(
        s_t=1,
        mask_posterior_fn=GumbelSoftMax_fn(),
        mask_prior_fn=ind_Bernoulli_fn(
            prior_prob=initial_prior_droprate, prob=prior_dropoutrate[0])
    )


    x = tf.constant([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                     [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])

    # x = tf.constant([[3.0, 3.0 , 3.0]])

    print(tf.keras.backend.get_value(f(x, training=False)))
    print(f'mean: {tf.keras.backend.get_value(f.mask_posterior.distribution.probs)}')
