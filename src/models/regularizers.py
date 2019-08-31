from tensorflow import keras


class L2RegularizerProvider:
    def __init__(self, l2):
        self.l2 = l2

    def get_regularizer(self):
        if self.l2 > 0:
            return keras.regularizers.l2(self.l2)
        else:
            return None
