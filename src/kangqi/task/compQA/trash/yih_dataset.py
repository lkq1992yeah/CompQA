import numpy as np

import sys

from kangqi.util.LogUtil import LogInfo

class YihDataset(Dataset):

    def __init__(self, data_path, batch_size):
        super(YihDataset, self).__init__()




def trigram_np_convert(input_np, word_dict, tri_list, word_tri_list):
    tri_size = len(tri_list)


if __name__ == '__main__':
    data_path = sys.argv[1]

    dict_fp = data_path + '/vocab'
    word_list = []
    word_dict = {}
    with open(dict_fp, 'r') as br:
        for line_idx, line in enumerate(br.readlines()):
            word = line.strip()
            word_dict[word] = len(word_list)
            word_list.append(word)
    word_size = len(word_list)
    LogInfo.logs('%d word loaded from vocab.', word_size)

    tri_dict = {}
    tri_list = []
    word_tri_list = []          # tri gram index for each word
    for wd_idx in range(len(word_list)):
        word_tri_list.append([])
    for wd_idx, wd in enumerate(word_list):
        pad_word = '#' + word + '#'
        for i in range(len(pad_word) - 3):
            tri = pad_word[i : i+3]
            if tri not in tri_dict:
                tri_dict[tri] = len(tri_list)
                tri_list.append(tri)
            word_tri_list[wd_idx].append(tri_dict[tri])
    tri_size = len(tri_dict)            
    LogInfo.logs('%d trigram converted from %d vocab.', tri_size, word_size)
                
    q_np = np.load(data_path + 'q.npy')
    data_size, q_max_len = np.shape(q_np)

    q_tri_np = 
