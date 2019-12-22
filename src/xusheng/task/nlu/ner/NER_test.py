import numpy as np
import tensorflow as tf

from config import config_train
from data import DataLoader, BatchGenerator, VocabularyLoader, LabelLoader
from NER_model import NER_basic, NER_self_attention


args = config_train()
print("Parameters:")
for attr, value in sorted(args.items(), reverse=True):
    print("{}={}".format(attr.upper(), value))
print("")

vocab_loader = VocabularyLoader()
vocab_loader.load_vocab(args['vocab_file'], args["embedding_dim"], args['encoding'])
args['vocab_size'] = vocab_loader.vocab_size
print("Embedding shape: %s.", vocab_loader.vocab_embedding.shape)

label_loader = LabelLoader()
label_loader.load_label(args['label_file'], args['encoding'])
#args['num_classes'] = label_loader.label_size
print("Number of labels: %d.", label_loader.label_size)

data_loader = DataLoader(args["max_seq_len"],
                         vocab_loader.vocab_index_dict,
                         label_loader.label_index_dict)
data_loader.load(args["data_file"], args['encoding'])

query_idx_test, query_len_test, _ = zip(*data_loader.data)

print("Create models...")
# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=True)
    session_conf.gpu_options.allow_growth = True

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # build model
        model = NER_basic(batch_size=args['batch_size'],
                          vocab_size=args['vocab_size'],
                          word_dim=args['embedding_dim'],
                          lstm_dim=args['lstm_dim'],
                          max_seq_len=args['max_seq_len'],
                          num_classes=args['num_classes'],
                          lr=args['learning_rate'],
                          gradient_clip=args['max_grad_norm'],
                          l2_reg_lambda=args['l2_reg_lambda'],
                          init_embedding=vocab_loader.vocab_embedding,
                          layer_size=args['layer_size'])

        sess.run(tf.global_variables_initializer())

        model.load(sess, "models/best_model")
        tag_list = model.decode(sess, np.array(query_idx_test),
                                np.array(query_len_test))


with open("results/test.txt", "w") as fout:
    for i in range(len(tag_list)):
        query_len = query_len_test[i]
        for j in range(query_len):
            fout.write(vocab_loader.index_vocab_dict[query_idx_test[i][j]])
            if j < query_len - 1:
                fout.write(" ")
        fout.write("\t")
        for j in range(query_len):
            fout.write(label_loader.index_label_dict[tag_list[i][j]])
            if j < query_len - 1:
                fout.write(" ")
        fout.write("\n")