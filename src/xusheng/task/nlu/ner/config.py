import argparse

def config_train():
    parser = argparse.ArgumentParser()

    # Data and vocabulary file
    parser.add_argument('--data_file', type=str,
                        help='all data file.')
    parser.add_argument('--vocab_file', type=str,
                        help='vocab embedding file.')
    parser.add_argument('--label_file', type=str,
                        help='label index file.')
    parser.add_argument('--encoding', type=str,
                        default='utf-8',
                        help='the encoding of the data file.')

    # Data format
    parser.add_argument('--max_seq_len', type=int,
                        help='max sequence length')
    parser.add_argument('--num_classes', type=int,
                        help='number of labels')
    parser.add_argument('--vocab_size', type=int,
                        help='size of vocab')
    parser.add_argument('--embedding_dim', type=int,
                        help='dimension of embedding')

    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='fraction of data used for training.')
    parser.add_argument('--valid_frac', type=float, default=0.1,
                        help='fraction of data used for validation.')
    # test_frac is computed as (1 - train_frac - valid_frac).
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='dropout keep rate, default to 1.0 (no dropout).')
    parser.add_argument('--eval_step', type=int, default=10,
                        help='every steps to evaluation.')

    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='initial learning rate')
    parser.add_argument('--l2_reg_lambda', type=float, default=0.0,
                        help='l2 reg lambda.')

    # Parameters for model.
    parser.add_argument('--lstm_dim', type=int,
                        help='lstm dimension.')
    parser.add_argument('--layer_size', type=int,
                        help='layer size.')

    args = parser.parse_args()

    return vars(args)