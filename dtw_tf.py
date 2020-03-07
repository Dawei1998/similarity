import tensorflow as tf
import numpy as np
import pandas as pd

# dataset (N*T), dataframe in pandas
# N: number of points
# T: len of series
# in this project, T = 24

def tf_l2norm_distmat_batch(X, Y, data_format='NWC'):
    '''
        X: [x1, x2, x3] (N,T,d) Y: [y1, y2, y3] (N,T)
        compute l2_norm
        return: [d(x1,y1), d(x2,y2), d(x3,y3)] (T*T,N)
    '''
    N, T = tf.shape(X)[0], tf.shape(X)[1]
    X = tf.reshape(tf.tile(X,[1,T]),(N*T*T,1))
    Y = tf.reshape(Y,(N*T,1))
    Y = tf.reshape(tf.tile(Y,[1,T]),(N*T*T,1))
    res = tf.squared_difference(X, Y)
    res = tf.sqrt(res)  # assume C=3
    res = tf.reshape(res, [N, T, T])  # [N, Tx, Ty]
    res = tf.transpose(res, (1, 2, 0))  # [Tx, Ty, N]
    res = tf.reshape(res, [T * T, N])  # [Tx*Ty, N]
    return res

def tf_dtw_batch(X, Y):
    '''
        X: [x1, x2, x3] (N,T) Y: [y1, y2, y3] (N,T)
        returns the accumulated matrix D
        D[len_x, len_y] is the DTW distance between x and y
    '''
    # X, Y : [N, T]
    dist_mats = tf_l2norm_distmat_batch(X, Y)  # [T*T, N]
    batch_size, max_time = tf.shape(X)[0], tf.shape(X)[1]
    d_array = tf.TensorArray(tf.float32, size=max_time * max_time, clear_after_read=False)
    d_array = d_array.unstack(dist_mats)  # read(t) returns an [N,] array at t timestep
    D = tf.TensorArray(tf.float32, size=(max_time + 1) * (max_time + 1), clear_after_read=False)
    # initalize
    def cond_boder(idx, res):
        return idx < max_time + 1
    def body_border_x(idx, res):
        res = res.write(tf.to_int32(idx * (max_time + 1)), 10000*tf.ones(shape=(batch_size,)))
        return idx + 1, res
    def body_border_y(idx, res):
        res = res.write(tf.to_int32(idx), 10000*tf.ones(shape=(batch_size,)))
        return idx + 1, res
    _, D = tf.while_loop(cond_boder, body_border_x, (1, D))
    _, D = tf.while_loop(cond_boder, body_border_y, (1, D))
    D = D.write(tf.to_int32(0),tf.zeros(shape=(batch_size,)))

    def cond(idx, res):
        return idx < (max_time + 1) * (max_time + 1)
    def body(idx, res):
        i = tf.to_int32(tf.divide(idx, max_time + 1))
        j = tf.mod(tf.to_int32(idx), max_time + 1)
        def f1():
            dt = d_array.read(i * (max_time + 1) + j - max_time - i - 1)
            min_v = tf.minimum(res.read((i - 1) * (max_time + 1) + j), res.read(i * (max_time + 1) + j - 1))
            min_v = tf.minimum(min_v, res.read((i - 1) * (max_time + 1) + j - 1))
            return res.write(idx, min_v + dt)
        def f2():
            return res
        res = tf.cond(tf.less(i, 1) | tf.less(j, 1),
                      true_fn=f2,
                      false_fn=f1)
        return idx + 1, res
    _, D = tf.while_loop(cond, body, (0, D))
    D = D.stack()
    D = tf.reshape(D, (max_time + 1, max_time + 1, batch_size))  # [T+1, T+1, N]
    return D

def tf_dtw(dataset, lens, batch_size = None):
    '''
        dataset [N, maxlen]
    '''
    N, T  = dataset.shape
    batch_size = batch_size or (N - 1) * N // 2
    batch_size = min(batch_size, (N - 1) * N // 2, 10000)
    idx_is, idx_js = [], []
    for i in range(N):
        for j in range(i + 1, N):
            idx_is.append(i)
            idx_js.append(j)
    #print(N, len(idx_is))
    for _ in range(batch_size - len(idx_is) + len(idx_is) // batch_size * batch_size):
        idx_is.append(0)
        idx_js.append(0)
    idx_is, idx_js = np.array(idx_is), np.array(idx_js)
    print(N, len(idx_is), batch_size)
    X = tf.placeholder(dtype=tf.float32, shape=(batch_size, T))
    Y = tf.placeholder(dtype=tf.float32, shape=(batch_size, T))
    D = tf_dtw_batch(X, Y)
    with tf.Session() as sess:
        res = []
        for i in range(len(idx_is) // batch_size):
            print("batch {}".format(i))
            cur_range = range(i * batch_size, (i + 1) * batch_size)
            cur_is, cur_js = idx_is[cur_range], idx_js[cur_range]
            feed_dict = {
                X: dataset.iloc[cur_is],
                Y: dataset.iloc[cur_js]
            }
            cur_D = sess.run(D, feed_dict=feed_dict)  # [T+1, T+1, bz]
            res.append([cur_D[lens[cur_is[j]], lens[cur_js[j]], j] for j in range(batch_size)])
    res = np.concatenate(res)
    dtw_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dtw_mat[i, j] = res[i * N + j - (i + 1) * (i + 2) // 2]
            dtw_mat[j, i] = dtw_mat[i, j]
    return dtw_mat

data = np.array([(0,0,0,0,0,0,0,0,0,0,0,3,1,0,0,0,3,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,2,0,0,0,0,1,0,2,1,2,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,2,0,0,0)])
ll = pd.DataFrame(data,index=['1','2','3'],columns=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
lens = len(ll)*[24]
result = tf_dtw(ll,lens)
sim = 1/(1+result)
print(result)
print(sim)  
