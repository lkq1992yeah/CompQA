"""
Training the single-task model
currently support NER (sequence labeling task)
"""

import numpy as np
import tensorflow as tf

from config import config_train
from data import DataLoader, BatchGenerator, VocabularyLoader, LabelLoader
from NER_model import NER_basic, NER_self_attention, \
    eval_seq_crf_with_o, eval_seq_crf_no_o


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
args['num_classes'] = label_loader.label_size
print("Number of labels: %d.", label_loader.label_size)

data_loader = DataLoader(args["max_seq_len"],
                         vocab_loader.vocab_index_dict,
                         label_loader.label_index_dict)
data_loader.load(args["data_file"], args['encoding'])

print("Create train, valid, test split...")
train_size = int(args["train_frac"] * data_loader.data_size)
valid_size = int(args["valid_frac"] * data_loader.data_size)
test_size = data_loader.data_size - train_size - valid_size

train_data = data_loader.data[:train_size]

query_idx_valid, query_len_valid, label_valid = \
    zip(*data_loader.data[train_size:train_size+valid_size])

query_idx_test, query_len_test, label_test = \
    zip(*data_loader.data[train_size+valid_size:])

print("train: valid: test = %d: %d: %d.", train_size, valid_size, test_size)

batch_generator = BatchGenerator(train_data, args["batch_size"])

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

        '''
        model = NER_self_attention(batch_size=args['batch_size'],
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
        '''

        sess.run(tf.global_variables_initializer())

        best_f1_valid, best_precision_valid, best_recall_valid = 0.0, 0.0, 0.0
        best_step = 0
        best_path = None
        stop = False

        for epoch in range(0, args["num_epochs"]):
            if stop:
                break
            print("Epoch %d/%d...", epoch, args["num_epochs"])
            batch_generator.reset_batch_pointer()
            for batch in range(batch_generator.num_batches):
                if stop:
                    break
                x_batch, seq_len_batch, y_batch = batch_generator.next_batch()
                current_step, loss = model.train_step(sess, x_batch, y_batch, seq_len_batch,
                                                      args['dropout_keep_prob'])

                print("Batch %d/%d ==> loss: %.4f",
                      batch+1, batch_generator.num_batches, loss)

                if current_step % args["eval_step"] == 0:
                    print("Eval on validation set...")
                    tag_list = model.decode(sess, np.array(query_idx_valid),
                                            np.array(query_len_valid))

                    precision_valid = eval_seq_crf_no_o(tag_list,
                                                        [x[:y] for x, y in zip(label_valid, query_len_valid)],
                                                        label_loader.index_label_dict,
                                                        method='precision')
                    recall_valid = eval_seq_crf_no_o(tag_list,
                                                     [x[:y] for x, y in zip(label_valid, query_len_valid)],
                                                     label_loader.index_label_dict,
                                                     method='recall')
                    if precision_valid == 0 or recall_valid == 0:
                        f1_valid = 0.0
                    else:
                        f1_valid = 2 * precision_valid * recall_valid / (precision_valid + recall_valid)

                    print("Precision-valid: %.4f, Recall-valid: %.4f, F1-valid: %.4f",
                          f1_valid, precision_valid, recall_valid)

                    # valid result improved, testing on test set
                    if f1_valid > best_f1_valid:
                        best_precision_valid = precision_valid
                        best_recall_valid = recall_valid
                        best_f1_valid = f1_valid
                        best_step = current_step
                        best_path = model.save(sess, "models")
                        print("Saved model checkpoint to {}\n".format(best_path))

                    if current_step - best_step > 2000:
                        print("Dev acc is not getting better in 2000 steps, triggers normal early stop")
                        stop = True

        print('-------------Show the results:--------------')
        print("Precision-valid: %.4f, Recall-valid: %.4f, F1-valid: %.4f",
              best_f1_valid, best_precision_valid, best_recall_valid)

        if best_path is None:
            print("No best model!!!!!!!")
        else:
            model.load(sess, best_path)
            tag_list = model.decode(sess, np.array(query_idx_test),
                                    np.array(query_len_test))
            precision_test = eval_seq_crf_no_o(tag_list,
                                               [x[:y] for x, y in zip(label_test, query_len_test)],
                                               label_loader.index_label_dict,
                                               method='precision')
            recall_test = eval_seq_crf_no_o(tag_list,
                                            [x[:y] for x, y in zip(label_test, query_len_test)],
                                            label_loader.index_label_dict,
                                            method='recall')
            if precision_test == 0 or recall_test == 0:
                f1_test = 0.0
            else:
                f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)

            print("Precision-test: %.4f, Recall-test: %.4f, F1-test: %.4f",
                  f1_test, precision_test, recall_test)