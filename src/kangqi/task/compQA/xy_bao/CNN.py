from basic_graph_gen import BasicQueryGraphGen
import tensorflow as tf
from tensorflow.contrib.keras.layers import Embedding, conv1D

VOCABULARY_SIZE = 1000
ENTITY_SIZE = 1000
MODEL_DIM = 1000

class siamCNN:
    def __init__(self):
        self.left = tf.placeholder(dtype=tf.int32)
        self.right = tf.placeholder(dtype=tf.int32)
        left_embed = Embedding(VOCABULARY_SIZE, MODEL_DIM)(self.left)
        right_embed = Embedding(ENTITY_SIZE, MODEL_DIM)(self.right)

        

def gen_intitial_data():
