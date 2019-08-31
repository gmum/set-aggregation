import itertools

import numpy as np
import tensorflow as tf


def normalize_image(example, eps=1e-4):
    example = np.array(example, dtype=np.float32)

    xs = example[:, 0]
    ys = example[:, 1]

    x_max, x_min = max(xs), min(xs)
    y_max, y_min = max(ys), min(ys)
    n, m = (x_max - x_min) / 2. + eps, (y_max - y_min) / 2. + eps

    example[:, 0] /= n
    example[:, 1] /= m

    example[:, 0] = example[:, 0] - 1 - x_min / n
    example[:, 1] = example[:, 1] - 1 - y_min / m

    return example


def iter_function(img):
    x = np.arange(img.shape[0].value, dtype=np.int32)
    y = np.arange(img.shape[1].value, dtype=np.int32)
    prod_ = list(itertools.product(x, y))
    img_values = tf.gather_nd(img, prod_)
    prod_ = normalize_image(prod_)
    return tf.concat([prod_, img_values], axis=1)


def prepare_batch(batch_x):
    return tf.map_fn(iter_function, batch_x)


def prepare_idx(col_idx, row_idx, col_max, row_max, EPS=1e-4):
    row_idx = tf.cast(row_idx,tf.float32)
    col_idx = tf.cast(col_idx,tf.float32)
    
    row_max = tf.cast(row_max, tf.float32)
    col_max = tf.cast(col_max, tf.float32)
    n, m = (row_max-1)/2, (col_max-1)/2
    row_idx = row_idx/n
    col_idx = col_idx/m
    
    row_idx = row_idx - 1
    col_idx = col_idx - 1
    return col_idx, row_idx



def prepare_batch_v2(batch_x):
    #can deal with None batches, assuming that within batch width and height is the same for all examples
    shape = tf.shape(batch_x)
    reshaped = tf.reshape(batch_x, [shape[0], shape[1]*shape[2], batch_x.shape[3]])
    #return reshaped
    r1 = tf.range(shape[1])
    r2 = tf.range(shape[2])
    col_idx = tf.reshape(tf.tile(r2,[shape[1]]),[shape[1],shape[2]])
    row_idx = tf.transpose(tf.reshape(tf.tile(r1,[shape[2]]),[shape[2],shape[1]]))
    col_idx, row_idx = prepare_idx(col_idx, row_idx, shape[2], shape[1])
    
    col_res = tf.reshape(col_idx,[-1]) 
    row_res = tf.reshape(row_idx,[-1])
    
    col_tile = tf.reshape(tf.tile(col_res, [shape[0]]),(shape[0],-1, 1))
    row_tile = tf.reshape(tf.tile(row_res, [shape[0]]),(shape[0],-1, 1))
    return tf.concat([row_tile, col_tile, reshaped], axis=2)    
