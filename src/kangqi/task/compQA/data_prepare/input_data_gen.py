# -*- coding: utf-8 -*-

#==============================================================================
# Author: Kangqi Luo
# Goal: Generate all the data (inputs of tensor) that can be DIRECTLY used in the model.
# Currently we focus on generating ordinal QA information only.
#==============================================================================

import sys

from .u import load_complex_questions, load_complex_questions_ordinal_only
from .loss_calc import LossCalculator
from .data_saving import DataSaver
from ..candgen.cand_gen import CandidateGenerator

from kangqi.util.config import load_configs
from kangqi.util.LogUtil import LogInfo
#from kangqi.util.cache import DictCache

class InputGenerator(object):

    def __init__(self, use_sparql_cache=True,
                 data_mode='Ordinal', sc_mode='Skeleton',
                 root_path='/home/kangqi/workspace/PythonProject',
                 cache_dir='runnings/compQA/cache'):
        LogInfo.begin_track('Initializing InputGenerator ... ')

        assert data_mode in ('Ordinal', 'ComplexQuestions')
        assert sc_mode in ('Skeleton', 'Sk+Ordinal')

        self.data_mode = data_mode
        self.sc_mode = sc_mode
        if self.data_mode == 'Ordinal':
            self.qa_data = load_complex_questions_ordinal_only()
            self.train_qa_list, self.test_qa_list = self.qa_data
        elif self.data_mode == 'ComplexQuestions':
            self.qa_data = load_complex_questions()
            self.train_qa_list, self.test_qa_list = self.qa_data
        else:
            LogInfo.logs('Unknown data mode: %s', self.data_mode)
        self.cand_gen = CandidateGenerator(use_sparql_cache=use_sparql_cache)
        self.loss_calc = LossCalculator(driver=self.cand_gen.driver)

#        qa_schema_score_cache_fp = '%s/%s/qa_schema_score_%s_cache' %(root_path, cache_dir, sc_mode)
#        self.score_cache = DictCache(qa_schema_score_cache_fp)
        LogInfo.end_track()

    # given each QA information, generate all the candidate schemas and calculate their score.
    def generate_schema_with_score(
            self, Tvt, min_surface_score, min_pop,
            use_ext_sk, min_ratio, vb=0):
        qa_schema_dict = {} # <qa, (schema, (lower_predict_set, score_dict))>
        qa_data = self.train_qa_list if Tvt == 'T' else self.test_qa_list
        qa_sz = len(qa_data)

#        curious_set = set([136, 162, 431, 577, 633, 767, 785, 854])

        for qa_idx in range(qa_sz):
#            if not (Tvt == 'T' and (qa_idx + 1) in curious_set):
#                continue

            qa = qa_data[qa_idx]
            q = qa.q
            qa.tokens = self.cand_gen.linker.parser.parse(q).tokens
            lower_answer_set = set(qa.lower_a_list)
            LogInfo.begin_track('Entering [%s] %d / %d QA: ', Tvt, qa_idx + 1, qa_sz)

#            schema_score_list = self.score_cache.get(qa)
#            if schema_score_list is None:

            schema_score_list = []
            LogInfo.logs(qa.to_string())
            LogInfo.logs('Token size = %d.', len(qa.tokens))

            LogInfo.begin_track('Work for schema generation ... ')
            schema_list = self.cand_gen.run_candgen(q,
                            min_surface_score=min_surface_score, min_pop=min_pop,
                            use_ext_sk=use_ext_sk, min_ratio=min_ratio, vb=vb)

            LogInfo.end_track('%d schemas extracted.', len(schema_list))

            filt_schema_list = []
            if self.sc_mode == 'Skeleton':
                for schema in schema_list:
                    if len(schema.constraints) == 0:
                        filt_schema_list.append(schema)
            elif self.sc_mode == 'Sk+Ordinal': # Just allow skeleton and ordinal constraints
                for schema in schema_list:
                    has_extra = False
                    for constr in schema.constraints:
                        if constr.constr_type in ('Entity', 'Type', 'Time'):
                            has_extra = True
                            break
                    if not has_extra:
                        filt_schema_list.append(schema)
            LogInfo.logs('%d schemas filtered [%s].', len(filt_schema_list), self.sc_mode)

            filt_sz = len(filt_schema_list)
            for sc_idx in range(filt_sz):
                if vb >= 1:
                    LogInfo.begin_track(
                        'Calculate loss for schema %d / %d: ',
                        sc_idx + 1, filt_sz)
                sc = filt_schema_list[sc_idx]
                if vb >= 1: sc.display()
                lower_predict_set, score_dict = \
                    self.loss_calc.calculate(sc, lower_answer_set, vb=vb)
                schema_score_list.append((sc, (lower_predict_set, score_dict)))
                # [(schema, (lower_predict_set, score_dict))]
                if vb >= 1: LogInfo.end_track()

#                self.score_cache.put(qa, schema_score_list)

            qa_schema_dict[qa] = schema_score_list

            # Show sorted information
            if vb >= 0:
                LogInfo.logs(qa.to_string())
                schema_info_list = qa_schema_dict[qa]
                srt_list = sorted(schema_info_list, lambda x, y: -cmp(x[1][1]['f1'], y[1][1]['f1']))
                for idx in range(min(len(srt_list), 20)):
                    sc, (lower_predict_set, score_dict) = srt_list[idx]
                    LogInfo.begin_track('Rank %d / %d: ', idx + 1, len(srt_list))
                    LogInfo.logs('score = %s', score_dict)
                    # LogInfo.logs('lower predict set = %s', lower_predict_set)
                    sc.display()
                    LogInfo.end_track()

            LogInfo.end_track('End of [%s] %d / %d QA.', Tvt, qa_idx + 1, qa_sz)

        return qa_schema_dict


if __name__ == '__main__':
    LogInfo.begin_track('[input_data_gen] starts ... ')
    try_dir = sys.argv[1]

    root_path = '/home/kangqi/workspace/PythonProject'
    input_dir = root_path + '/runnings/compQA/input/%s' %try_dir
    LogInfo.logs('Input directory saved to "%s".', input_dir)

    config_fp = input_dir + '/param_config'
    config_dict = load_configs(config_fp)

    use_sparql_cache = True if config_dict['use_sparql_cache'] == 'True' else False
    data_mode = config_dict['data_mode']
    sc_mode = config_dict['sc_mode']
    use_ext_sk = True if config_dict['use_ext_sk'] == 'True' else False
    min_ratio = float(config_dict['min_ratio'])
    min_surface_score = float(config_dict['min_surface_score'])
    min_pop = float(config_dict['min_pop'])

    data_saver = DataSaver(
                   sent_max_len=int(config_dict['sent_max_len']),
                   n_word_emb=int(config_dict['n_word_emb']),
                   path_max_len=int(config_dict['path_max_len']),
                   n_path_emb=int(config_dict['n_path_emb']),
                   w2v_fp=config_dict['w2v_fp'])
    input_gen = InputGenerator(
                    use_sparql_cache=use_sparql_cache,
                    data_mode=data_mode,
                    sc_mode=sc_mode)

    for Tvt in ('T', 't'):
        LogInfo.begin_track('Input generation for %s: ', Tvt)

        LogInfo.begin_track('Generating schema & loss for %s-data: ', Tvt)
        qa_schema_dict = input_gen.generate_schema_with_score(Tvt=Tvt,
                min_surface_score=min_surface_score,
                min_pop=min_pop, use_ext_sk=use_ext_sk,
                min_ratio=min_ratio, vb=0)
        LogInfo.end_track('Schema & loss generated for %d %s-data.', len(qa_schema_dict), Tvt)

        tokens_size_list = []
        schema_size_list = []
        effective_schema_size_list = []   # schemas with F1 > 0

        for qa, schema_score_list in qa_schema_dict.items():
            tokens_size_list.append(len(qa.tokens))
            schema_size_list.append(len(schema_score_list))
            effective = 0
            for schema, (lower_predict_set, score_dict) in schema_score_list:
                if score_dict['f1'] > 0.0:
                    effective += 1
            effective_schema_size_list.append(effective)

        import numpy as np
        tokens_size_list.sort()
        schema_size_list.sort()
        effective_schema_size_list.sort()
        tokens_size_arr = np.asarray(tokens_size_list, dtype='int')
        schema_size_arr = np.asarray(schema_size_list, dtype='int')
        effective_schema_size_arr = np.asarray(effective_schema_size_list, dtype='int')

        np.set_printoptions(threshold=np.inf)
        LogInfo.logs('Tokens: mean = %f, std = %f, values = %s.',
                     tokens_size_arr.mean(), tokens_size_arr.std(), tokens_size_arr)
        LogInfo.logs('Schema: mean = %f, std = %f, values = %s.',
                     schema_size_arr.mean(), schema_size_arr.std(), schema_size_arr)
        LogInfo.logs('Effective Schema: mean = %f, std = %f, values = %s.',
                     effective_schema_size_arr.mean(),
                     effective_schema_size_arr.std(),
                     effective_schema_size_arr)
        np.set_printoptions(threshold=1000)

        PN = int(config_dict['PN'])
        LogInfo.logs('PN = %d', PN)
        save_fp = '%s/%s_input.pydump' %(input_dir, Tvt)
        data_saver.save_qa_data(qa_schema_dict, PN, save_fp, Tvt, sc_mode=sc_mode)

        LogInfo.end_track()

    LogInfo.end_track()







