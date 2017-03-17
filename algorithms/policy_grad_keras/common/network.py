from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, TFOptimizer
import tensorflow as tf     # needs to use tf optimizer instead of Keras one
import numpy as np


def make_full_network(config):
    network = AgentPolicyNN(config)
    return network.prepare_loss().compute_gradients()


def make_shared_network(config):
    network = GlobalPolicyNN(config)
    return network.apply_gradients()


# Simple 2-layer fully-connected Policy Neural Network
class GlobalPolicyNN(object):
    # This class is used for global-NN and holds only weights on which applies computed gradients
    def __init__(self, config):
        self.global_t = K.variable(0, 'int64')    # tf.Variable(0, tf.int64)
        self.increment_global_t = K.update_add(self.global_t, 1)  # tf.assign_add(self.global_t, 1)

        self._input_size = np.prod(np.array(config.state_size))
        self._action_size = config.action_size

        if type(config.layers_size) not in [list, tuple]:
            config.layers_size = [config.layers_size]

        # Define keras model immediately --> we just define vars in tf case and its "links" for agent later.
        # We also can define variables manually, but work directly with models is more keras-like style.
        input_ = Input(shape=(self._input_size,))
        throw_ = Dense(config.layers_size[0], activation='elu', init='glorot_normal')(input_)

        # glorot_uniform == xavier, we can also set: bias_initializer='zeros'
        for i in range(1, len(config.layers_size)):
            throw_ = Dense(config.layers_size[i], activation='elu', init='glorot_normal')(throw_)

        throw_ = Dense(self._action_size, activation='softmax', init='glorot_normal')(throw_)
        self.net = Model(input=input_, output=throw_)

        self.values = self.net.trainable_weights
        '''
        self.values = [tf.get_variable('W0', shape=[self._input_size, config.layers_size[0]],
                                       initializer=tf.contrib.layers.xavier_initializer())]
        idx = len(config.layers_size)
        for i in range(1, idx):
            self.values.append(tf.get_variable('W%d' % i, shape=[config.layers_size[i-1], config.layers_size[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
        self.values.append(tf.get_variable('W%d' % idx, shape=[config.layers_size[-1], self._action_size],
                                           initializer=tf.contrib.layers.xavier_initializer()))
        '''
        self._placeholders = [K.placeholder(v.get_shape(), dtype=v.dtype) for v in self.values]
        self._assign_values = tf.group(*[
            K.update(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.gradients = [K.placeholder(v.get_shape(), dtype=v.dtype) for v in self.values]
        self.learning_rate = config.learning_rate

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })

    def get_vars(self):
        return self.values

    def apply_gradients(self):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate
        )
        self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, self.values))
        return self


class AgentPolicyNN(GlobalPolicyNN):
    # This class additionally implements loss computation and gradients wrt this loss
    def __init__(self, config):
        super(AgentPolicyNN, self).__init__(config)

        # state (input)
        self.s = self.net.input

        # policy (output)
        self.pi = self.net.output

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]
        # return self.net(s_t)

    def compute_gradients(self):
        self.grads = K.gradients(self.loss, self.values)
        return self

    def prepare_loss(self):
        self.a = K.placeholder((None, self._action_size), dtype='float32', name="taken_action")
        self.advantage = K.placeholder(ndim=2, dtype='float32', name="discounted_reward")

        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.
        log_like = K.log(K.sum(tf.multiply(self.a, self.pi)))
        self.loss = -K.mean(log_like * self.advantage)

        return self
