import numpy as np
from tensorflow import keras

class DataLoader:
    def __init__(self, dataname):
        self.dataname = dataname

    @staticmethod
    def __part(corpus, corpus_labels, train_size, test_size, valid_size, shuffle, random_state):
        if train_size + valid_size + test_size > len(corpus):
            raise ValueError("test_size+valid_size+train_size cannot be greater then the whole corpus")
        if shuffle:
            rs = np.random.RandomState(random_state)
            rs.shuffle(corpus)
            rs = np.random.RandomState(random_state)
            rs.shuffle(corpus_labels)
        print(train_size, test_size, valid_size, "part")
        valid_data, test_data, train_data = DataLoader.__get_partition_sets(corpus, test_size, train_size, valid_size)
        valid_labels, test_labels, train_labels = DataLoader.__get_partition_sets(corpus_labels, test_size, train_size,
                                                                                  valid_size)
        print(len(train_data),len(valid_data), len(test_data))
        return (train_data, valid_data, test_data), (train_labels, valid_labels, test_labels)

    @staticmethod
    def __get_partition_sets(corpus, test_size, train_size, valid_size):
        valid_data = corpus[:valid_size]
        train_data = corpus[valid_size:valid_size + train_size]
        test_data = corpus[valid_size + train_size:valid_size + train_size + test_size]
        return valid_data, test_data, train_data

    @staticmethod
    def get_data_sizes_from_ratio(corpus_length, valid_ratio, test_ratio):
        valid_size = int(corpus_length * valid_ratio)
        test_size = int(corpus_length * test_ratio)
        train_size = corpus_length - (valid_size + test_size)
        return train_size, valid_size, test_size

    @staticmethod
    def get_data(dataset, dataset_labels, valid_ratio, test_ratio, shuffle=True, random_state=10):
        print(len(dataset))
        if len(dataset) != len(dataset_labels):
            raise ValueError("The labels size and the data size do not much")
        train_size, valid_size, test_size = DataLoader.get_data_sizes_from_ratio(len(dataset), valid_ratio,
                                                                                   test_ratio)
        print(train_size,valid_size,test_size)
        return DataLoader.__part(dataset, dataset_labels, train_size, test_size, valid_size, shuffle=shuffle,
                                 random_state=random_state)

    def load(self, shuffle, random_state):
        pass

    def is_raw(self):
        pass

class CifarDataLoader(DataLoader):
    def __init__(self, dataname="cifar", valid_ratio=None, test_ratio=None, raw=False):
        super().__init__(dataname)
        self.valid_ratio = 0 if valid_ratio is None else valid_ratio
        self.test_ratio = 0 if test_ratio is None else test_ratio
        self.raw = raw

    def is_raw(self):
        return self.raw

    def load(self, shuffle, random_state):
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        if not self.raw:
            X_train = X_train / 255.
            X_test = X_test / 255.
        self.dataset = np.concatenate((X_train, X_test))
        self.labels = np.concatenate((y_train, y_test)).flatten()
        return DataLoader.get_data(self.dataset, self.labels, self.valid_ratio, self.test_ratio, shuffle, random_state)


class FashionMnistDataLoader(DataLoader):
    def __init__(self,dataname="fashion-mnist", valid_ratio=None, test_ratio=None, raw=False):
        super().__init__(dataname)
        self.valid_ratio = 0 if valid_ratio is None else valid_ratio
        self.test_ratio = 0 if test_ratio is None else test_ratio
        self.raw = raw

    def is_raw(self):
        return self.raw

    def load(self, shuffle, random_state):
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        if not self.raw:
            X_train = X_train / 255.
            X_test = X_test /255.
        self.dataset = np.concatenate((X_train, X_test)).reshape((len(X_train)+len(X_test),X_train.shape[1], X_train.shape[2], 1))
        print(self.dataset.shape)
        self.labels = np.concatenate((y_train, y_test)).flatten()
        return DataLoader.get_data(self.dataset, self.labels, self.valid_ratio, self.test_ratio, shuffle, random_state)


class NLPDataLoader(DataLoader):
    def __init__(self, dataname, num_words=None):
        super().__init__(dataname)
        self.num_words = num_words

    def load(self, shuffle, random_state):
        pass

    def get_word_index(self):
        pass

    def get_index_word(self):
        word_index = self.get_word_index()
        return dict([(value, key) for (key, value) in word_index.items()])

    def is_raw(self):
        return False
