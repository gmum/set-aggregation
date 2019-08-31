import tensorflow as tf


class HeadBuilder:
    def get_head(self, input, labels, scope, reuse):
        pass


class SimpleCrossEntropyHeadBuilder(HeadBuilder):
    def __init__(self, output_dim, hidden_dim, regularizer_provider, batch_norm = False, dropout=0.0, layers=0, is_training=True):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.r_provider = regularizer_provider
        self.layers = layers
        self.batch_norm = batch_norm
        self.is_training=is_training

    def get_head(self, input, labels, scope, reuse=tf.AUTO_REUSE):
        self.labels_one_hot = tf.one_hot(labels, self.output_dim)
        with tf.variable_scope(scope, reuse=reuse):

            if self.batch_norm:
                output = tf.layers.batch_normalization(input, training=self.is_training)
                input = output
            if self.dropout > 0.0:
                output = tf.layers.dropout(input, rate=self.dropout, training=self.is_training)
                input = output

            self.hidden_layers = []
            out = input
            for i in range(self.layers):
                out = tf.layers.Dense(self.hidden_dim, kernel_regularizer=self.r_provider.get_regularizer())(out)
                self.hidden_layers.append(out)
                if self.batch_norm:
                    out = tf.layers.batch_normalization(out, training=self.is_training)
                if self.dropout > 0.0:
                    out = tf.layers.dropout(out, rate=self.dropout, training=self.is_training)
            pred = tf.layers.Dense(self.output_dim, kernel_regularizer=self.r_provider.get_regularizer())(out)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_one_hot, logits=pred)
            loss = tf.reduce_mean(loss)
            return loss, pred
