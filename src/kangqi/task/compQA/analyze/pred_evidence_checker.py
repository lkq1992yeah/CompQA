from ..dataset.dataset_acl18 import SchemaDatasetACL18
from ..dataset.u import load_compq
from ..util.word_emb import WordEmbeddingUtil

from kangqi.util.LogUtil import LogInfo


def main():
    qa_list = load_compq()

    LogInfo.begin_track('Loading Utils ... ')
    wd_emb_util = WordEmbeddingUtil(wd_emb='glove', dim_emb=300)
    wd_emb_util.load_word_indices()
    wd_emb_util.load_mid_indices()
    LogInfo.end_track()

    LogInfo.begin_track('Creating Dataset ... ')
    schema_dataset = SchemaDatasetACL18(
        data_name='CompQ', data_dir='runnings/candgen_CompQ/180209_ACL18_SMART',
        file_list_name='all_list', q_max_len=20, sc_max_len=3, path_max_len=3, item_max_len=5,
        schema_level='strict', wd_emb_util=wd_emb_util, verbose=1
    )
    LogInfo.end_track()
    schema_dataset.load_smart_cands()

    preds_support_dict = {}     # <predicates, [(q_idx, sc_path_display, max_rm_f1)]>

    for q_idx in range(len(qa_list)):
        cand_list = schema_dataset.smart_q_cand_dict[q_idx]
        dedup_dict = {}
        for sc in cand_list:
            key = sc.get_rm_key()
            dedup_dict.setdefault(key, []).append(sc)
        for key, local_sc_list in dedup_dict.items():
            max_rm_f1 = max([sc.f1 for sc in local_sc_list])
            rep_sc = local_sc_list[0]
            path_str_list = [str(path) for path in rep_sc.path_list]
            related_preds = set([])     # all 1-hop 2-hop predicates from the current rm_sc
            for pred_path in rep_sc.path_list:
                if len(pred_path) == 1:
                    related_preds.add(pred_path[0])
                else:
                    related_preds.add(pred_path[0])
                    related_preds.add(pred_path[1])
                    related_preds.add(pred_path[0] + '-->' + pred_path[1])
            if max_rm_f1 > 0 and q_idx < 1000:    # only collect positive detail support for training questions.
                for preds in related_preds:
                    preds_support_dict.setdefault(preds, []).append((
                        q_idx, rep_sc.ori_idx, key, path_str_list, max_rm_f1))

    all_preds = preds_support_dict.keys()
    all_preds.sort()
    for preds in all_preds:
        LogInfo.begin_track('Checking [%s]:', preds)
        support_tup_list = preds_support_dict[preds]
        good_size = len(filter(lambda _tup: _tup[-1] >= 0.1, support_tup_list))
        LogInfo.logs('Found %3d supports with F1 > 0.', len(support_tup_list))
        LogInfo.logs('Found %3d supports with F1 >= 0.1.', good_size)
        support_tup_list.sort(key=lambda _tup: _tup[-1], reverse=True)
        for support_idx, tup in enumerate(support_tup_list):
            q_idx, ori_idx, key, path_str_list, max_rm_f1 = tup
            LogInfo.begin_track('Support %d / %d:', support_idx+1, len(support_tup_list))
            LogInfo.logs('Q-%04d: [%s]', q_idx, qa_list[q_idx]['utterance'].encode('utf-8'))
            for path in path_str_list:
                LogInfo.logs('        %s', path)
            LogInfo.logs('rm_F1 = %.6f', max_rm_f1)
            LogInfo.logs('sc_line = %d', ori_idx)
            # LogInfo.logs('key = %s', key)
            LogInfo.end_track()
        LogInfo.end_track()


if __name__ == '__main__':
    main()
