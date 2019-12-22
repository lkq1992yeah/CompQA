# -*- coding:utf-8 -*-

import os
import numpy as np
from kangqi.util.LogUtil import LogInfo


class KBEmbeddingUtil(object):

    @staticmethod
    def mid_transform(mid):
        # /people/person/place_of_birth --> people.person.place_of_birth
        if not mid.startswith('/'):
            return mid
        return mid[1:].replace('/', '.')

    def __init__(self, kb_emb_dir, dim_kb_emb, verbose=0):
        self.kb_emb_dir = kb_emb_dir
        self.dim_kb_emb = dim_kb_emb
        self.verbose = verbose

        self.e_idx_dict = self.p_idx_dict = None
        self.e_emb_matrix = self.p_emb_matrix = None

        # self.n_entities = self.n_preds = None

    def load_entity_ids(self):
        if self.e_idx_dict is None:
            self.e_idx_dict = self.load_ids('entity')

    def load_predicate_ids(self):
        if self.p_idx_dict is None:
            self.p_idx_dict = self.load_ids('relation')
            offset = len(self.p_idx_dict)
            for pred, p_idx in self.p_idx_dict.items():
                rev_pred = '!' + pred
                rev_p_idx = p_idx + offset
                self.p_idx_dict[rev_pred] = rev_p_idx

    def load_entity_embeddings(self):
        if self.e_emb_matrix is None:
            self.e_emb_matrix = self.load_embeddings('entity')
            # self.n_entities = self.e_emb_matrix.shape[0]

    def load_predicate_embeddings(self):
        if self.p_emb_matrix is None:
            p_emb_matrix = self.load_embeddings('relation')
            if p_emb_matrix is not None:
                rev_p_emb_matrix = -1. * p_emb_matrix
                self.p_emb_matrix = np.concatenate([p_emb_matrix, rev_p_emb_matrix], axis=0)
                # self.n_preds = self.p_emb_matrix.shape[0]
            else:
                self.p_emb_matrix = None

    def load_ids(self, item_name):
        item_id_fp = '%s/%s2id.txt' % (self.kb_emb_dir, item_name)
        item_idx_dict = {}
        with open(item_id_fp, 'r') as br:
            size = int(br.readline().strip())
            pairs = br.readlines()
            assert size == len(pairs)
            for pair in pairs:
                spt = pair.strip().split('\t')
                item = self.mid_transform(spt[0])
                idx = int(spt[1])
                item_idx_dict[item] = idx
        LogInfo.logs('%d <%s, idx> loaded from [%s].', len(item_idx_dict), item_name, item_id_fp)
        return item_idx_dict

    def load_embeddings(self, item_name):
        item_emb_txt_fp = '%s/%s2vec_%d.txt' % (self.kb_emb_dir, item_name, self.dim_kb_emb)
        item_emb_npy_fp = '%s/%s2vec_%d.npy' % (self.kb_emb_dir, item_name, self.dim_kb_emb)

        if os.path.isfile(item_emb_npy_fp):
            LogInfo.logs('Loading %s embedding from BINARY [%s] ...', item_name, item_emb_npy_fp)
            item_emb_matrix = np.load(item_emb_npy_fp)
            LogInfo.logs('%s %s embedding loaded.', item_emb_matrix.shape, item_name)
        elif os.path.isfile(item_emb_txt_fp):
            LogInfo.logs('Loading %s embedding from TXT [%s] ...', item_name, item_emb_txt_fp)
            item_emb_list = []
            with open(item_emb_txt_fp, 'r') as br:
                lines = br.readlines()
                for line_idx, line in enumerate(lines):
                    emb = map(lambda x: float(x), line.strip().split('\t'))
                    item_emb_list.append(emb)
            item_emb_matrix = np.array(item_emb_list, dtype='float32')
            LogInfo.logs('%s %s embedding loaded.', item_emb_matrix.shape, item_name)
            LogInfo.logs('Saving binary to [%s] ...', item_emb_npy_fp)
            np.save(item_emb_npy_fp, item_emb_matrix)
            LogInfo.logs('Saved.')
        else:       # no corresponding txt file, we just leave it blank
            LogInfo.logs('Do not load %s embedding, not found [%s].', item_name, item_emb_txt_fp)
            item_emb_matrix = None

        if item_emb_matrix is not None:
            assert item_emb_matrix.shape[1] == self.dim_kb_emb
        return item_emb_matrix

    def get_entity_idx(self, entity):
        return self.get_item_idx(entity, self.e_idx_dict)

    def get_predicate_idx(self, pred):
        return self.get_item_idx(pred, self.p_idx_dict)

    @staticmethod
    def get_item_idx(mid, item_dict):
        idx = item_dict.get(mid, -1)
        if idx == -1:               # missing items, add to the dict
            idx = len(item_dict)
            item_dict[mid] = idx
        return idx
