from .base_schema_dl_builder import BaseSchemaDLBuilder
from .input_feat_gen_helper import InputFeatureGenHelper

from kangqi.util.LogUtil import LogInfo


class DepSchemaDLBuilder(BaseSchemaDLBuilder):

    def __init__(self, schema_dataset, compq_mt_model, neg_pick_config, parser_port, dep_or_cp):
        # neg_pick_config = {'neg_f1_ths': neg_f1_ths,
        #                    'neg_max_sample': neg_max_sample}
        LogInfo.logs('DepSchemaDLBuilder initializing ...')
        BaseSchemaDLBuilder.__init__(self, schema_dataset=schema_dataset,
                                     compq_mt_model=compq_mt_model,
                                     neg_pick_config=neg_pick_config)
        self.feat_gen_helper = InputFeatureGenHelper(schema_dataset=schema_dataset, parser_port=parser_port)
        self.dep_or_cp = dep_or_cp
        LogInfo.logs('Dependency or ContextPattern: [%s]', dep_or_cp)
        assert dep_or_cp in ('dep', 'cp')

    """
    A statistics of CompQ: on average, 1.22 schemas share the same RM structures.
    If entity type information makes difference to the dependency parsing step,
    then we must put detail MID into the key representation.
    However, as a simpler solution, for now we just put [st, ed) into it, instead of the detail mid.   
    """
    def get_rm_key(self, sc):
        # category, start, end, detail_path
        # Just copy from kq_schema.get_rm_key(), and no need to change "Main" into "Entity"
        rep_list = []
        for raw_path, using_pred_seq in zip(sc.raw_paths, sc.path_list):
            category, gl_data, _ = raw_path
            """ entity position matters ! """
            local_rep = '%s:%s:%s:%s' % (category, gl_data.start, gl_data.end, '|'.join(using_pred_seq))
            rep_list.append(local_rep)
        rep_list.sort()
        return '\t'.join(rep_list)

    """
    Current don't care the actual type of an entity in the schema
    """
    def get_el_key(self, sc):
        # start, end, mid (E only, ignore T/Tm/Ord)
        rep_list = []
        for raw_path in sc.raw_paths:
            category, gl_data, pred_seq = raw_path
            if category not in ('Main', 'Entity'):
                continue
            start = gl_data.start
            end = gl_data.end
            mid = gl_data.value
            mid_pos_repr = '%d:%d:%s' % (start, end, mid)
            rep_list.append(mid_pos_repr)
        rep_list.sort()
        return '\t'.join(rep_list)

    def get_full_key(self, sc):
        return self.get_el_key(sc) + '\t' + self.get_rm_key(sc)

    def input_feat_gen(self, sc, is_in_train, **kwargs):
        if sc.input_np_dict is not None:
            return

        extra_feats = self.feat_gen_helper.generate_extra_feat(sc=sc)
        dep_input, dep_len = self.feat_gen_helper.generate_dep_feat(sc=sc, dep_or_cp=self.dep_or_cp)
        qw_input, qw_len = self.feat_gen_helper.generate_qw_feat__same(sc=sc, is_in_train=is_in_train)
        el_indv_feats, el_comb_feats, el_mask = self.feat_gen_helper.generate_el_feat(sc=sc)
        (path_size, path_cates,
         path_ids, pw_input, pw_len,
         pseq_ids, pseq_len) = self.feat_gen_helper.generate_whole_path_feat(sc=sc)
        sc.input_np_dict = {
            'path_size': path_size,
            'path_cates': path_cates,
            'path_ids': path_ids,
            'pw_input': pw_input,
            'pw_len': pw_len,
            'pseq_ids': pseq_ids,
            'pseq_len': pseq_len,

            'qw_input': qw_input,
            'qw_len': qw_len,
            'dep_input': dep_input,
            'dep_len': dep_len,

            'el_indv_feats': el_indv_feats,
            'el_comb_feats': el_comb_feats,
            'el_mask': el_mask,
            'extra_feats': extra_feats
        }
