import numpy as np
from sqlalchemy import false
import tensorflow as tf
import selfies as sf
import warnings
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import IPythonConsole
import os
from src.utils.logger import GLogger

from src.core.oracle_base import Oracle
from src.dataset.data_instance_base import DataInstance
from src.dataset.dataset_base import Dataset



class TfGCNOracle(Oracle):

    def __init__(self, id, oracle_store_path, config_dict=None, n_node_types=118) -> None:
        super().__init__(id, oracle_store_path, config_dict)
        self.name = 'gcn_tf'
        self._clf = None
        self._n_node_types = n_node_types
        self._max_n_nodes = 0


    def create_model(self):
        # code modified ///////////////////////////////////////
        # ninput = tf.keras.Input(
        #     (
        #         None,
        #         100,
        #     )
        # )
        ninput = tf.keras.Input(
            (
                None,
                self._n_node_types,
            )
        )
        # /////////////////////////////////////////////////////

        ainput = tf.keras.Input(
            (
                None,
                None,
            )
        )

        # GCN block
        x = GCNLayer("relu")([ninput, ainput])
        x = GCNLayer("relu")(x)
        x = GCNLayer("relu")(x)
        x = GCNLayer("relu")(x)
        # reduce to graph features
        x = GRLayer()(x)
        # standard layers
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        gcnmodel = tf.keras.Model(inputs=(ninput, ainput), outputs=x)

        # Compile the model
        gcnmodel.compile(
            "adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        self._clf = gcnmodel
        

    def fit(self, dataset: Dataset, split_i=0):

        self._n_node_types = dataset.n_node_types
        self._max_n_nodes = dataset.max_n_nodes
        self.create_model()

        # Creating the name of the folder for storing the trained oracle
        oracle_name = self.name + '_fit_on_' + dataset.name

        # Creating the rute to store the oracle
        oracle_uri = os.path.join(self._oracle_store_path, oracle_name)

        if os.path.exists(oracle_uri):
            # Load the weights of the trained model
            self.name = oracle_name
            self.read_oracle(oracle_name)

        else:
            # Create the folder to store the oracle if it does not exist
            os.mkdir(oracle_uri)        
            self.name = oracle_name

            # wrap the data in tensorflow format
            # code modified ////////////////////////////////////////////////////////////
            # data = tf.data.Dataset.from_generator(
            #     dataset.gen_tf_data,
            #     output_types=((tf.float32, tf.float32), tf.float32),
            #     output_shapes=(
            #         (tf.TensorShape([None, 100]), tf.TensorShape([None, None])),
            #         tf.TensorShape([]),
            #     ),
            # )
            data = tf.data.Dataset.from_generator(
                dataset.gen_tf_data,
                output_types=((tf.float32, tf.float32), tf.float32),
                output_shapes=(
                    (tf.TensorShape([None, self._n_node_types]), tf.TensorShape([None, None])),
                    tf.TensorShape([]),
                ),
            )
            # //////////////////////////////////////////////////////////////////////////

            if split_i == -1:
                # train with the entire dataset
                val_data, train_data = data, data.shuffle(1000)
            else:
                # Choose the data splits
                N = len(dataset.instances)
                split = int(0.1 * N)
                test_data = data.take(split)
                nontest = data.skip(split)
                val_data, train_data = nontest.take(split), nontest.skip(split).shuffle(1000)

            # class_weight = {0: 1.0, 1: 30.0}  # to account for class imbalance
            # Fit the model
            result = self._clf.fit(
                train_data.batch(128),
                validation_data=val_data.batch(128),
                callbacks=[MyCallback()],
                epochs=30,
                verbose=0
                # class_weight=class_weight
            )

            # Storing the trained oracle
            self.write_oracle()


    def _real_predict(self, data_instance):
        nodes, adj_mat = data_instance.to_numpy_arrays(false, self._max_n_nodes, self._n_node_types)
        pred = self._clf((nodes[np.newaxis, ...], adj_mat[np.newaxis, ...])).numpy()
        labels = np.array(pred).flatten()
        bin_labels = np.where(labels > 0.5, np.ones(len(labels)), np.zeros(len(labels)))
        return int(np.array(bin_labels).flatten())

    def _real_predict_proba(self, data_instance):
        pred = self._real_predict(data_instance)
        if pred :
            return [0,1]
        return [1,0]
    
    def embedd(self, instance):
        return instance


    def write_oracle(self):
        # Creating the rute to store the oracle
        oracle_uri = os.path.join(self._oracle_store_path, self.name, 'oracle.ckpt')

        # Save the weights
        self._clf.save_weights(oracle_uri)


    def read_oracle(self, oracle_name):
        # Creating the rute to stored oracle
        oracle_uri = os.path.join(self._oracle_store_path, oracle_name, 'oracle.ckpt')

        # loading the stored weights
        self._clf.load_weights(oracle_uri)


class GCNLayer(tf.keras.layers.Layer):
    """Implementation of GCN as layer"""

    def __init__(self, activation=None, **kwargs):
        # constructor, which just calls super constructor
        # and turns requested activation into a callable function
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # create trainable weights
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name="w")

    def call(self, inputs):
        # split input into nodes, adj
        nodes, adj = inputs
        # compute degree
        degree = tf.reduce_sum(adj, axis=-1)
        # GCN equation
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)
        return out, adj


class GRLayer(tf.keras.layers.Layer):
    """Reduction layer: A GNN layer that computes average over all node features"""

    def __init__(self, name="GRLayer", **kwargs):
        super(GRLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        nodes, adj = inputs
        reduction = tf.reduce_mean(nodes, axis=1)
        return reduction


class MyCallback(tf.keras.callbacks.Callback):

    def __init__(self, patience=0):
        super(MyCallback, self).__init__()
        self._logger = GLogger.getLogger()
        self._logger.info("Created %s",str(self.__class__))

    def on_epoch_begin(self, epoch, logs=None):
        self._logger.info(f'Oracle training epoch {epoch}')