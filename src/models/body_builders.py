from src.data_handling.utils import *
from src.layers.projection_layers import projection_layer


class BodyBuilder:
    def get_body(self, input, scope, priorities, weights, reuse=tf.AUTO_REUSE):
        pass


class ImageCNNFeatureBuilder(BodyBuilder):
    def __init__(self, kernel_sizes, filter_nums, max_poolings, regularizer_provider, batch_norm, dropout, is_training):
        self.kernel_sizes = kernel_sizes
        self.filter_nums = filter_nums
        self.max_poolings = max_poolings
        self.r_provider = regularizer_provider
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.is_training = is_training
        print(batch_norm, dropout)

    def get_feature(self, input, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            output = input
            num = 0
            for kernel, filters, max_pool in zip(self.kernel_sizes, self.filter_nums, self.max_poolings):
                output = tf.layers.conv2d(inputs=output, filters=filters, kernel_size=kernel, padding='SAME',
                                          activation=tf.nn.relu,
                                          name="conv_{}".format(num),
                                          kernel_regularizer=self.r_provider.get_regularizer())
                num += 1
                if self.batch_norm:
                    output = tf.layers.batch_normalization(inputs=output, axis=3, training=self.is_training)
                if max_pool:
                    output = tf.layers.max_pooling2d(inputs=output, pool_size=2, strides=2, padding='SAME')
                if self.dropout > 0.0:
                    output = tf.layers.dropout(inputs=output, rate=self.dropout, training=self.is_training)
                
        return output


class BaselineCNNBodyBuilder(ImageCNNFeatureBuilder):
    def __init__(self, kernel_sizes, filter_nums, max_poolings, regularizer_provider,
                 pooling="max", batch_norm=False, dropout=0.0, is_training=True):
        super().__init__(kernel_sizes, filter_nums, max_poolings, regularizer_provider, batch_norm, dropout, is_training)
        self.pooling = pooling

    def get_body(self, input, scope, priorities=None, weights=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            output = self.get_feature(input, scope)
            if self.pooling == "max":
                output = tf.keras.layers.GlobalMaxPooling2D()(output)
            elif self.pooling == "avg":
                output = tf.keras.layers.GlobalAveragePooling2D()(output)
            elif self.pooling == "flatten":
                output = tf.layers.flatten(output)
            else:
                raise ValueError("Unknown pooling type {}".format(self.pooling))
        return output


class Conv1x1CNNBodyBuilder(ImageCNNFeatureBuilder):
    def __init__(self, kernel_sizes, filter_nums, max_poolings, regularizer_provider, batch_norm=False, dropout=0.0,
                 is_training=True):
        super().__init__(kernel_sizes, filter_nums, max_poolings, regularizer_provider, batch_norm, dropout, is_training)

    def get_body(self, input, scope, priorities=None, weights=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            output = self.get_feature(input, scope)

            output = tf.layers.conv2d(inputs=output, filters=32, kernel_size=1, padding="SAME",
                                      activation=tf.nn.relu, name="conv1x1",
                                      kernel_regularizer=self.r_provider.get_regularizer())

            output = tf.layers.flatten(output)
        return output


class SetCNNBodyBuilder(ImageCNNFeatureBuilder):
    def __init__(self, kernel_sizes, filter_nums, max_poolings, projection_dim, regularizer_provider, batch_norm=False,
                 dropout=0.0, is_training=True, reduce_type="mean"):
        super().__init__(kernel_sizes, filter_nums, max_poolings, regularizer_provider, batch_norm, dropout, is_training)
        self.projection_dim = projection_dim
        self.reduce_type = reduce_type

    def get_body(self, input, scope, priorities=None, weights=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            output = self.get_feature(input, scope)
            x_conv = prepare_batch(output)
        output = projection_layer(x_conv, weights, self.projection_dim, scope, reduce_type=self.reduce_type, reuse=reuse)
        return output


class PlainSetBodyBuilder(BodyBuilder):
    def __init__(self, regularizer_provider, projection_dim):
        super().__init__()
        self.r_provider = regularizer_provider
        self.projection_dim = projection_dim

    def get_body(self, input, scope, priorities, weights, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.concat([input, priorities], axis=2)
            output = projection_layer(out, weights, self.projection_dim, regularizer=self.r_provider.get_regularizer())
        return output


class MaxPoolBodyBuilder(BodyBuilder):
    def __init__(self):
        super().__init__()

    def get_body(self, input, scope, priorities=None, weights=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            output = tf.reduce_max(input, axis=1)
        return output


class AvgPoolBodyBuilder(BodyBuilder):
    def __init__(self):
        super().__init__()

    def get_body(self, input, scope, priorities=None, weights=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            output = tf.reduce_mean(input, axis=1)
        return output




