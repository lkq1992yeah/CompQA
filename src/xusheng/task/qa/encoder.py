
import tensorflow as tf
from xusheng.task.qa.schema import Node, Edge, Schema
from xusheng.model.rnn_encoder import BidirectionalRNNEncoder
from xusheng.model.attention import AttentionLayerBahdanau, AttentionLayerAvg

class SchemaEncoder(object):

    def __init__(self):
        pass

    def encode_schema(self, schema, config):
        """
        Encode a schema to hidden representation by 
        combining several sub-paths starting from 
        subj node or constraint nodes.
        :param schema: input schema
        :param config: parameter dict
        :return: hidden representation with shape [dim, ]
        """
        U = tf.get_variable(name="U",
                            shape=[config["transE_dim"], config["hidden_dim"]],
                            dtype=tf.float32,
                            initializer
                            =tf.contrib.layers.xavier_initializer(uniform=True))
        V = tf.get_variable(name="V",
                            shape=[config["transE_dim"], config["hidden_dim"]],
                            dtype=tf.float32,
                            initializer
                            =tf.contrib.layers.xavier_initializer(uniform=True))
        W = tf.get_variable(name="W",
                            shape=[config["hidden_dim"] * config["max_len"], config["output_dim"]],
                            dtype=tf.float32,
                            initializer
                            =tf.contrib.layers.xavier_initializer(uniform=True))
        b = tf.get_variable(name="b",
                            shape=[config["output_dim"]],
                            dtype=tf.float32,
                            initializer
                            =tf.constant_initializer(0.0))
        paths = list()
        for node in schema.constraints:
            state = tf.constant(node.embedding)
            while node.skeleton_edge is not None:
                x = node.skeleton_edge.embedding
                state = tf.nn.relu(tf.add(tf.matmul(state, U), tf.matmul(x, V)))
                node = node.skeleton_node
            paths.append(state)
        concat = paths[0]
        for i in range(1, config["max_len"]):
            if i >= len(paths):
                concat = tf.concat([concat, tf.constant(0.0, shape=[config["hidden_dim"], ])],
                                   axis=-1)
            else:
                concat = tf.concat([concat, paths[i]], axis=-1)
        output = tf.nn.relu(tf.add(b, tf.matmul(concat, W)))
        return output

    def encode_question(self, question, question_len, answer, config):
        """
        Encode question with answer-aware attention
        :param question: [B, T, dim]
        :param question_len: [B, T, ]
        :param answer: [B, dim]
        :param config: parameter dict
        :return: [B, hidden_dim]
        """

        # bi-LSTM
        with tf.name_scope("rnn_encoder"):
            rnn_config = dict()
            key_list = ["cell_class", "num_units", "dropout_input_keep_prob",
                        "dropout_output_keep_prob", "num_layers", "reuse"]
            for key in key_list:
                rnn_config[key] = config[key]
            rnn_encoder = BidirectionalRNNEncoder(rnn_config, config["mode"])
            encoder_output = rnn_encoder.encode(question, question_len)

        # attention mechanism
        with tf.name_scope("attention"):
            att_config = dict()
            key_list = ["num_units"]
            for key in key_list:
                att_config[key] = config[key]

            if config["attention"] == "bah":
                att = AttentionLayerBahdanau(att_config)
                question_hidden = att.build(answer,
                                            encoder_output.attention_values,
                                            encoder_output.attention_values_length)
            elif config["attention"] == "avg":
                att = AttentionLayerAvg()
                question_hidden = att.build(encoder_output.attention_values,
                                            encoder_output.attention_values_length)

        return question_hidden

