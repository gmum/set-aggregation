import numpy as np
import cv2
import warnings

class DataProvider():
    def __init__(self, DataLoader):
        self.loader = DataLoader
        self.loaded = False

    def load_dataset(self, shuffle=True, random_state=10):
        data, labels = self.loader.load(shuffle, random_state)
        self.train_data, self.valid_data, self.test_data = data
        self.train_labels, self.valid_labels, self.test_labels = labels
        self.loaded = True
        self.random_state = np.random.RandomState(seed=random_state)

    def get_dataset(self, dataset_type):
        assert self.loaded, "Data must be loaded before using this method"
        if dataset_type == "train":
            return self.train_data, self.train_labels
        elif dataset_type == "valid":
            return self.valid_data, self.valid_labels
        elif dataset_type == "test":
            return self.test_data, self.test_labels
        else:
            raise ValueError("Unknown dataset type name (allowed: train, valid, test)")

    def get_random_batch(self, batch_size, dataset_type="train"):
        assert self.loaded, "Data must be loaded before using this method"
        dataset, dataset_labels = self.get_dataset(dataset_type)
        idx = np.random.choice(np.arange(len(dataset)), batch_size, replace=False)
        X, y = dataset[idx], dataset_labels[idx]
        return X, y, None, None

    def get_next_batch(self, batch_size, iter, dataset_type="train"):
        assert self.loaded, "Data must be loaded before using this method"
        dataset, dataset_labels = self.get_dataset(dataset_type)
        X = dataset[iter * batch_size:iter * batch_size + batch_size]
        y = dataset_labels[iter * batch_size:iter * batch_size + batch_size]
        return X, y, None, None

    def valid_len(self):
        assert self.loaded, "Data must be loaded before using this method"
        return len(self.valid_data)

    def test_len(self):
        assert self.loaded, "Data must be loaded before using this method"
        return len(self.test_data)

    def __len__(self):
        assert self.loaded, "Data must be loaded before using this method"
        return len(self.train_data)

    def get_dataset_length(self,datatype):
        assert self.loaded, "Data must be loaded before using this method"
        if datatype == "train":
           return len(self.train_data)
        elif datatype == "valid":
           return len(self.valid_data)
        elif datatype == "test":
           return len(self.test_data)
        else:
           raise ValueError("Unknown datatype")

    def __shuffle(self, data, labels, int_seed):
        random_state = np.random.RandomState(seed=int_seed)
        random_state.shuffle(data)
        random_state = np.random.RandomState(seed=int_seed)
        random_state.shuffle(labels)

    def shuffle(self, dataset_type, seed_high=100000):
        assert self.loaded, "Data must be loaded before using this method"
        seed = self.random_state.randint(seed_high)

        if dataset_type == "train":
            self.__shuffle(self.train_data,self.train_labels,seed)
        elif dataset_type == "valid":
            self.__shuffle(self.valid_data, self.valid_labels,seed)
        elif dataset_type == "test":
            self.__shuffle(self.test_data, self.test_labels, seed)
        else:
            raise ValueError("Unknown dataset type name (allowed: train, valid, test)")

class RandomSizeDataProvider(DataProvider):
    def __init__(self, DataLoader, scale_seed=123):
        super().__init__(DataLoader)
        self.shapes = [20,24,28,32,36,40,44,48,52,56]
        self.scale_state = np.random.RandomState(scale_seed)


    def __rescale(self, input):
        i = self.scale_state.randint(len(self.shapes))
        res = np.array(list(map(lambda x: cv2.resize(x, dsize=(self.shapes[i], self.shapes[i]), interpolation=cv2.INTER_CUBIC), input)))
        if len(res.shape)==3:
            #happens for gray scale images, cv2 just cuts the last dim.
            res = res.reshape((res.shape[0],res.shape[1],res.shape[2],1))
        return res/255.


    def load_dataset(self, shuffle=True, random_state=10):
        assert self.loader.is_raw(), "data loader must provide raw, int data images"
        data, labels = self.loader.load(shuffle, random_state)
        self.train_data, self.valid_data, self.test_data = data
        self.train_labels, self.valid_labels, self.test_labels = labels
        self.loaded = True

    def get_dataset_length(self,dataset_type):
        assert self.loaded, "Data must be loaded"
        if dataset_type == "train":
            return len(self.train_labels)
        elif dataset_type == "valid":
            return len(self.valid_labels)
        elif dataset_type == "test":
            return len(self.test_labels)
        else:
            raise ValueError("Unknown dataset type name: {}".format(dataset_type))


    def get_random_batch(self, batch_size, dataset_type="train", same=False):
        assert self.loaded, "Data must be loaded before using this method"
        dataset, dataset_labels = self.get_dataset(dataset_type)
        idx = np.random.choice(np.arange(len(dataset)), batch_size, replace=False)
        X, y = dataset[idx], dataset_labels[idx]
        if not same:
            X = self.__rescale(X)
        return X, y, None, None

    def get_next_batch(self, batch_size, iter, dataset_type="train", same=False):
        assert self.loaded, "Data must be loaded before using this method"
        dataset, dataset_labels = self.get_dataset(dataset_type)
        X = dataset[iter * batch_size:iter * batch_size + batch_size]
        y = dataset_labels[iter * batch_size:iter * batch_size + batch_size]
        if not same:
            X = self.__rescale(X)
        return X, y, None, None



class MultiSizeDataProvider(DataProvider):
    def __init__(self, DataLoader, shapes):
        super().__init__(DataLoader)
        self.shapes = shapes


    def __rescale(self, input, i):
        res = np.array(list(map(lambda x: cv2.resize(x, dsize=(self.shapes[i], self.shapes[i]), interpolation=cv2.INTER_CUBIC), input)))
        if len(res.shape)==3:
            #happens for gray scale images, cv2 just cuts the last dim. 
            res = res.reshape((res.shape[0],res.shape[1],res.shape[2],1))
        return res/255.

    def __get_shapes_indicies(self):
        train_idxs = []
        valid_idxs = []
        test_idxs = []

        train_step_size = len(self.train_data) // len(self.shapes)
        valid_step_size = len(self.valid_data) // len(self.shapes)
        test_step_size = len(self.test_data) // len(self.shapes)

        for i in range(len(self.shapes)-1):
            train_idxs.append((i * train_step_size, (i+1) * train_step_size))
            valid_idxs.append((i * valid_step_size, (i + 1) * valid_step_size))
            test_idxs.append((i * test_step_size, (i + 1) * test_step_size))

        i = len(self.shapes)-1
        train_idxs.append((i * train_step_size, len(self.train_data)))
        valid_idxs.append((i * valid_step_size, len(self.valid_data)))
        test_idxs.append((i * test_step_size, len(self.test_data)))

        return train_idxs, valid_idxs, test_idxs

    def get_shape_dataset_length(self,datatype, i):
        if datatype == "train":
           return self.train_idxs[i][1]-self.train_idxs[i][0]+1
        elif datatype == "valid":
           return self.valid_idxs[i][1]-self.valid_idxs[i][0]+1
        elif datatype == "test":
           return self.test_idxs[i][1]-self.test_idxs[i][0]+1
        else:
           raise ValueError("Unknown datatype")

    def load_dataset(self, shuffle=True, random_state=10):
        assert self.loader.is_raw(), "data loader must provide raw, int data images"
        data, labels = self.loader.load(shuffle, random_state)
        self.train_data, self.valid_data, self.test_data = data
        self.train_labels, self.valid_labels, self.test_labels = labels
        self.train_idxs, self.valid_idxs, self.test_idxs = self.__get_shapes_indicies()
        self.loaded = True

    def get_dataset(self, dataset_type):
        warnings.warn("Warning: This function returns whole, unscaled dataset. For scaled daatset "
                      "use the rescale method with proper dataset input")
        assert self.loaded, "Data must be loaded before using this method"
        if dataset_type == "train":
            return self.train_data, self.train_labels
        elif dataset_type == "valid":
            return self.valid_data, self.valid_labels
        elif dataset_type == "test":
            return self.test_data, self.test_labels
        else:
            raise ValueError("Unknown dataset type name (allowed: train, valid, test)")

    def get_dataset_idxs(self, dataset_type):
        assert self.loaded, "Data must be loaded before using this method"
        if dataset_type == "train":
            return self.train_idxs
        elif dataset_type == "valid":
            return self.valid_idxs
        elif dataset_type == "test":
            return self.test_idxs
        else:
            raise ValueError("Unknown dataset type name (allowed: train, valid, test)")

    def get_random_batch(self, batch_size, dataset_type="train"):
        assert self.loaded, "Data must be loaded before using this method"
        dataset, dataset_labels = self.get_dataset(dataset_type)
        batch_step = batch_size//len(self.shapes)
        dataset_idxs = self.get_dataset_idxs(dataset_type)
        X = []
        y = []

        diff_size = batch_size-(len(self.shapes)-1)*batch_step
        diff_size_shape = np.random.randint(len(self.shapes))

     
        for i in range(len(self.shapes)):
            idx_bounds = dataset_idxs[i]
            size = diff_size if diff_size_shape == i else batch_step
            idx = np.random.choice(np.arange(idx_bounds[0],idx_bounds[1]), size, replace=False)
            X.append(self.__rescale(dataset[idx],i))
            y.append(dataset_labels[idx])

        return X, y, None, None

    def get_next_batch(self, batch_size, iter, dataset_type="train"):
        assert self.loaded, "Data must be loaded before using this method"
        dataset, dataset_labels = self.get_dataset(dataset_type)
        batch_step = batch_size // len(self.shapes)
        dataset_idxs = self.get_dataset_idxs(dataset_type)
        X = []
        y = []

        for i in range(len(self.shapes)):
            idx_bounds = dataset_idxs[i]
            begin = idx_bounds[0]+iter*batch_step
            end = min(idx_bounds[0]+(iter+1)*batch_step, idx_bounds[1])
            Xs = dataset[begin:end]
            ys = dataset_labels[begin:end]
            X.append(self.__rescale(Xs,i))
            y.append(ys)

        return X, y, None, None

    def __shuffle(self, data, labels, int_seed):
        raise NotImplementedError()

    def shuffle(self, dataset_type, seed_high=100000):
        raise NotImplementedError()
