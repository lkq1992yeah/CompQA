import sys
import codecs


def load_ours_meta(our_meta_fp):
    meta_dict = {}
    with codecs.open(our_meta_fp, 'r', 'utf-8') as br:
        lines = br.readlines()
        for line in lines:
            spt = line.strip().split('\t')
            if len(spt) != 11:
                continue
            if spt[1] == 'Q_idx':
                continue
            idx = int(spt[1])
            cand_size = int(spt[2])     # strict
            max_f1 = float(spt[6])
            meta_dict[idx] = {'utterance': spt[10], 'cand_size': cand_size, 'max_f1': max_f1}
    return meta_dict


def compare(dict1, dict2, bw):
    assert len(dict1) == len(dict2)
    tup_list = []
    for q_idx in dict1:
        tup_list.append((
            q_idx,
            dict1[q_idx]['cand_size'],
            dict1[q_idx]['max_f1'],
            dict2[q_idx]['cand_size'],
            dict2[q_idx]['max_f1'],
            dict1[q_idx]['max_f1'] - dict2[q_idx]['max_f1'],
            dict1[q_idx]['utterance']
        ))
    tup_list.sort(key=lambda _tup: _tup[-2])

    for rank_idx, tup in enumerate(tup_list):
        q_idx, c1, f1, c2, f2, delta, utt = tup
        bw.write('Rank-%04d\t%04d\t%4d\t%.6f\t%4d\t%.6f\t%.6f\t%s\n' % (
            rank_idx, q_idx, c1, f1, c2, f2, delta, utt))


def main(data_dir):
    meta_fp_1 = data_dir + '/log.schema_check'
    meta_fp_2 = 'runnings/candgen_CompQ/180209_ACL18_SMART/log.schema_check'
    out_fp = data_dir + '/log.upbound_comparison'
    with codecs.open(out_fp, 'w', 'utf-8') as bw:
        bw.write('Meta_1: %s\n' % meta_fp_1)
        bw.write('Meta_2: %s\n' % meta_fp_2)
        compare(load_ours_meta(meta_fp_1), load_ours_meta(meta_fp_2), bw)


if __name__ == '__main__':
    main(data_dir=sys.argv[1])
