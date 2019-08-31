import tensorflow as tf


def projection_layer(input, weights, projection_dim, scope, activation="relu", regularizer=None, reduce_type="mean", reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        if activation == 'relu':
            proj_conv = tf.layers.conv1d(input, projection_dim, 1, activation=activation, name="projection")
        else:
            raise ValueError("Unsupported activation")
        if weights is not None:
            if reduce_type=="mean":
                return tf.reduce_mean(weights * proj_conv, axis=(1))
            elif reduce_type=="sum":
                return tf.reduce_sum(weights * proj_conv, axis=(1))
            else:
                raise ValueError("Unknown reduce type")
        else:
            if reduce_type=="mean":
                return tf.reduce_mean(proj_conv, axis=(1))
            elif reduce_type=="sum":
                return tf.reduce_sum(proj_conv, axis=(1))
            else:
                raise ValueError("Unknown reduce type")
