import time
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf
import numpy as np
from deepctr_torch.models import DeepFM

# Original https://github.com/cheungdaven/DeepRec/blob/master/models/rating_prediction/autorec.py

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


def RMSE(error, num):
    return np.sqrt(error / num)


def MAE(error_mae, num):
    return (error_mae / num)


def transform_long_table_to_sparse_matrix(df, test_size):
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1]
        i = line[2]
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1])
        test_col.append(line[2])
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items


def train_test_autorec(rating_df, **kwargs):
    train_matrix, test_matrix, n_users, n_items = transform_long_table_to_sparse_matrix(rating_df, 0.1)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        model = IAutoRec(sess, n_users, n_items, epoch=50)
        model.build_network()
        print("Network built")
        log = model.execute(train_matrix, test_matrix)
    return model, log


class IAutoRec():
    def __init__(self, sess, num_user, num_item, learning_rate=0.001, reg_rate=0.1, epoch=500, batch_size=500,
                 verbose=False, T=3, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        print("IAutoRec.")

    def build_network(self, hidden_neuron=500):
        self.rating_matrix = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.rating_matrix_mask = tf.placeholder(dtype=tf.float32, shape=[self.num_user, None])
        self.keep_rate_net = tf.placeholder(tf.float32)
        self.keep_rate_input = tf.placeholder(tf.float32)

        V = tf.Variable(tf.random_normal([hidden_neuron, self.num_user], stddev=0.01))
        W = tf.Variable(tf.random_normal([self.num_user, hidden_neuron], stddev=0.01))

        mu = tf.Variable(tf.random_normal([hidden_neuron], stddev=0.01))
        b = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
        layer_1 = tf.nn.dropout(tf.sigmoid(tf.expand_dims(mu, 1) + tf.matmul(V, self.rating_matrix)),
                                self.keep_rate_net)
        self.layer_2 = tf.matmul(W, layer_1) + tf.expand_dims(b, 1)
        self.loss = tf.reduce_mean(tf.square(
            tf.norm(tf.multiply((self.rating_matrix - self.layer_2), self.rating_matrix_mask)))) + self.reg_rate * (
        tf.square(tf.norm(W)) + tf.square(tf.norm(V)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_data):
        self.num_training = self.num_item
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering

        for i in range(total_batch):
            start_time = time.time()
            if i == total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.rating_matrix: self.train_data[:, batch_set_idx],
                                               self.rating_matrix_mask: self.train_data_mask[:, batch_set_idx],
                                               self.keep_rate_net: 0.95
                                               })
            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self, test_data):
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.rating_matrix: self.train_data,
                                                                     self.rating_matrix_mask: self.train_data_mask,
                                                                     self.keep_rate_net: 1})
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict(u, i)
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse, mae = RMSE(error, len(test_set)), MAE(error_mae, len(test_set))
        print("RMSE:" + str(rmse) + "; MAE:" + str(mae))
        return rmse, mae

    def execute(self, train_data, test_data):
        self.train_data = self._data_process(train_data)
        self.train_data_mask = scipy.sign(self.train_data)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        errors_log = [] 
        for epoch in range(self.epochs):
            if self.verbose:
                print("Epoch: %04d;" % (epoch))
            self.train(train_data)
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch), end='')
                rmse, mae = self.test(test_data)
                errors_log.append({"epoch": epoch, "rmse": rmse, "mae": mae})
        return errors_log

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.reconstruction[user_id, item_id]

    def _data_process(self, data):
        output = np.zeros((self.num_user, self.num_item))
        for u in range(self.num_user):
            for i in range(self.num_item):
                output[u, i] = data.get((u, i))
        return output


class DeepFmModel:
    def __init__(self, linear_feature_columns, dnn_feature_columns, feature_names):
        self._linear_feature_columns = linear_feature_columns
        self._dnn_feature_columns = dnn_feature_columns
        self._feature_names = feature_names
        self._deepfm = DeepFM(
            self._linear_feature_columns,
            self._dnn_feature_columns,
            task='multiclass',
            device='cpu'
        )
        self._deepfm.compile("adam", "mse", metrics=['mse'], )
        
    def train(self, train_set, target_values):
        train_model_input = {n: train_set[n] for n in self._feature_names}
        history = self._deepfm.fit(
            train_model_input,
            target_values,
            batch_size=256,
            epochs=10,
            verbose=2,
            validation_split=0.2
        )

        return history

    def predict(self, test_set):
        test_model_input = {n: test_set[n] for n in self._feature_names}
        result = self._deepfm.predict(test_model_input, batch_size=256)
        return result
