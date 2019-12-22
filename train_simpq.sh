#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
PY='python -u -m';  QA=kangqi.task.compQA

DATA_NAME=SimpQ
DS_NAME=FB2M_Gold
DATA_DIR=runnings/candgen_${DATA_NAME}/${DS_NAME}
DATA_PREF=all;  FILE_LIST_NAME=${DATA_PREF}_list
SC_LEVEL=strict
DEP_SIMULATE=True

WD_EMB=glove;   DIM_EMB=300

NUM_UNITS=150               # 32, 64, [*150]
RNN_CELL=GRU                # None, [*GRU]
if [ "${RNN_CELL}" = "None" ]; then
    RNN_REPR=None
else
    RNN_REPR=${RNN_CELL}${NUM_UNITS}
fi

SCN_LAYERS=2
SCN_DROP=0.0
SCN_RESIDUAL=False
SCN_REPR=SCN${SCN_LAYERS}-d${SCN_DROP}-res${SCN_RESIDUAL}

W_EMB_FIX=Upd               # [Upd], Fix
PATH_USAGE=BH               # [X|B|R][X|B|R|H]   ([sumBOW], sumRNN, pwBOW, pwRNN, pOnly)
ATT_FUNC=noAtt              # [*noAtt], dot, bahdanau
DIM_ATT_HIDDEN=128          
SEQ_MERGE_MODE=fwbw         # [*fwbw], avg, max
SENT_USAGE=mSum             # qwOnly, depOnly, [*mSum], mMax, mWsum
SCORING_MODE=compact        # [*compact], separated, bao
FINAL_FUNC=cos              # [cos], dot, bilinear, fcxx

MARGIN=0.5;     LOSS=H${MARGIN}
OPTM_NAME=Adam; LEARNING_RATE=1e-3; BATCH=32; KEEP_PROB=1.0

NEG_F1_THS=0.100
NEG_MAX_SAMPLE=20        # 10, 20, 40, 80
NEG_STRATEGY=Fix         # Fix, Dyn
NEG_REPR=N${NEG_STRATEGY}-${NEG_MAX_SAMPLE}

FULL_OPTM_METHOD=rm
FULL_BP=False
PRETRAIN=10
if [ "${FULL_BP}" = "False" ]; then
    FBP_REPR=False
else
    FBP_REPR=${FULL_BP}${PRETRAIN}
fi

# RM_SUB=${RNN_REPR}_${W_EMB_FIX}_${PATH_USAGE}_${ATT_FUNC}_${SEQ_MERGE_MODE}_${SENT_USAGE}_${SCORING_MODE}_${FINAL_FUNC}
RM_SUB=${RNN_REPR}_${PATH_USAGE}_${SENT_USAGE}
SUB1=180519_${SC_LEVEL}
SUB2=${DATA_PREF}__${FULL_OPTM_METHOD}__${DS_NAME}__ds${DEP_SIMULATE}__drop0.2
SUB3=${RM_SUB}
SUB3=${NEG_REPR}__${SUB3}__b${BATCH}__fbp${FBP_REPR}

OUTPUT_DIR=runnings/${DATA_NAME}/${SUB1}/${SUB2}/${SUB3}
mkdir -p ${OUTPUT_DIR}
cp train_simpq.sh ${OUTPUT_DIR}/exec.sh

$PY $QA.main_dep \
    --machine Blackhole \
    --gpu_fraction 0.2 \
    --word_emb ${WD_EMB} \
    --dim_emb 300 \
    --dep_simulate ${DEP_SIMULATE} \
    --data_config "{'data_name': '${DATA_NAME}', \
                    'data_dir': '${DATA_DIR}', \
                    'file_list_name': '${FILE_LIST_NAME}', \
                    'schema_level': '${SC_LEVEL}', \
                   }" \
    --neg_pick_config "{'neg_f1_ths': ${NEG_F1_THS}, \
                        'neg_max_sample': ${NEG_MAX_SAMPLE}, \
                        'strategy': '${NEG_STRATEGY}', \
                        'cool_down': 1.0 \
                       }" \
    --model_config "{ \
        'rnn_config': { \
            'cell_class': '${RNN_CELL}', \
            'num_units': ${NUM_UNITS}, \
            'keep_prob': ${KEEP_PROB} \
        }, \
        'scn_config': { \
            'n_layers': ${SCN_LAYERS}, \
            'dropout': ${SCN_DROP}, \
            'residual': ${SCN_RESIDUAL}, \
        }, \
        'att_config': { \
            'dim_att_hidden': ${DIM_ATT_HIDDEN}, \
            'att_func': '${ATT_FUNC}', \
        }, \
        'w_emb_fix': '${W_EMB_FIX}', \
        'path_usage': '${PATH_USAGE}', \
        'seq_merge_mode': '${SEQ_MERGE_MODE}', \
        'sent_usage': '${SENT_USAGE}', \
        'scoring_mode': '${SCORING_MODE}', \
        'final_func': '${FINAL_FUNC}', \
        'loss_func': '${LOSS}', \
        'optm_name': '${OPTM_NAME}', \
        'learning_rate': ${LEARNING_RATE}, \
        'full_back_prop': ${FULL_BP}, \
    }" \
    --full_optm_method ${FULL_OPTM_METHOD} \
    --pre_train_steps ${PRETRAIN} \
    --max_epoch 20 \
    --max_patience 20 \
    --optm_batch_size ${BATCH} \
    --eval_batch_size 512 \
    --output_dir ${OUTPUT_DIR} \
    --save_best \
    --verbose 1 \
    > ${OUTPUT_DIR}/log.txt

