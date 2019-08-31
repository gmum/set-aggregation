import tensorflow as tf


class PlaceholderBuilder:
    def set_up_placeholders(self):
        pass


class ImagePlaceholderBuilder(PlaceholderBuilder):
    def __init__(self, width, height, channels):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height

    def set_up_placeholders(self):
        x = tf.placeholder(tf.float32, [None, self.width, self.height, self.channels], name="inputs")
        priorities = None
        weights = None
        y = tf.placeholder(tf.int64, [None], name="labels")
        return x, y, priorities, weights

class MultiImagePlaceholderBuilder(PlaceholderBuilder):
    def __init__(self, shapes, channels):
        super().__init__()
        self.shapes = shapes
        self.channels = channels

    def set_up_placeholders(self):
        x_placeholders = []
        priorities = []
        weights = []
        y_placeholders = []
        for s in self.shapes:
            x = tf.placeholder(tf.float32, [None, s, s, self.channels], name="inputs")
            x_placeholders.append(x)
            priorities.append(None)
            weights.append(None)
            y = tf.placeholder(tf.int64, [None], name="labels")
            y_placeholders.append(y)
        return x_placeholders, y_placeholders, priorities, weights

