"""
misc utils fro data processing & analysis & visualization
"""

from task.nlu.data import fuzzy_match_name
from util.log_util import LogInfo

import codecs
import copy


class EntityAdder(object):

    def __init__(self):
        self.root_fp = "/u01/xusheng/data"
        self.entity_set = set()
        self.stop_word_set = set()
        self.load_entity("kg_pinlei_id")
        self.load_entity("kg_pname_id")
        self.load_entity("kg_vname_id")
        self.load_stop_word("stopword_kg")

    def load_entity(self, type):
        LogInfo.begin_track("Load entity names...")
        with codecs.open(self.root_fp + "/raw/" + type, 'r',
                         encoding='utf-8') as fin:
            for line in fin:
                spt = line.strip().split("^A")
                if len(spt) < 2:
                    continue
                entity = spt[0]
                self.entity_set.add(entity)
        LogInfo.end_track("%s (total %d names) loaded.",
                          type, len(self.entity_set))

    def load_stop_word(self, type):
        LogInfo.begin_track("Load stop words...")
        with codecs.open(self.root_fp + "/raw/" + type, 'r',
                         encoding='utf-8') as fin:
            for line in fin:
                spt = line.strip().split("^A")
                if len(spt) < 2:
                    continue
                word = spt[0]
                self.stop_word_set.add(word)
        LogInfo.end_track("%s (total %d words) loaded.",
                          type, len(self.stop_word_set))

    def add_entity_tag_yyh(self):
        LogInfo.begin_track("Begin adding tags for pls...")
        fin = codecs.open(self.root_fp + "/yyh_w2v_train.txt",
                          'r',
                          encoding='utf-8')
        fout = codecs.open(self.root_fp + "/yyh_w2v_train.txt.entity_tag",
                           'w',
                           encoding='utf-8')
        cnt = 0
        for line in fin:
            spt = line.strip().split()
            new_line = ""
            i = 0
            while i < len(spt):
                if i + 6 < len(spt):
                    str7 = spt[i] + spt[i + 1] + spt[i + 2] + spt[i + 3] + \
                           spt[i + 4] + spt[i + 5] + spt[i + 6]
                    if str7 in self.entity_set and str7 not in self.stop_word_set:
                        LogInfo.logs("Found 7-term entity [%s|%s|%s|%s|%s|%s|%s]",
                                     spt[i], spt[i + 1], spt[i + 2], spt[i + 3],
                                     spt[i + 4], spt[i + 5], spt[i + 6])
                        new_line += "[[" + str7 + "]] "
                        i += 7
                        continue
                if i + 5 < len(spt):
                    str6 = spt[i] + spt[i + 1] + spt[i + 2] + spt[i + 3] + \
                           spt[i + 4] + spt[i + 5]
                    if str6 in self.entity_set and str6 not in self.stop_word_set:
                        LogInfo.logs("Found 6-term entity [%s|%s|%s|%s|%s|%s]",
                                     spt[i], spt[i + 1], spt[i + 2], spt[i + 3],
                                     spt[i + 4], spt[i + 5])
                        new_line += "[[" + str6 + "]] "
                        i += 6
                        continue
                if i + 4 < len(spt):
                    str5 = spt[i] + spt[i + 1] + spt[i + 2] + spt[i + 3] + spt[i + 4]
                    if str5 in self.entity_set and str5 not in self.stop_word_set:
                        LogInfo.logs("Found 5-term entity [%s|%s|%s|%s|%s]",
                                     spt[i], spt[i + 1], spt[i + 2], spt[i + 3], spt[i + 4])
                        new_line += "[[" + str5 + "]] "
                        i += 5
                        continue
                if i + 3 < len(spt):
                    str4 = spt[i] + spt[i + 1] + spt[i + 2] + spt[i + 3]
                    if str4 in self.entity_set and str4 not in self.stop_word_set:
                        # LogInfo.logs("Found 4-term entity [%s|%s|%s|%s]",
                        #              spt[i], spt[i + 1], spt[i + 2], spt[i + 3])
                        new_line += "[[" + str4 + "]] "
                        i += 4
                        continue
                if i + 2 < len(spt):
                    str3 = spt[i] + spt[i + 1] + spt[i + 2]
                    if str3 in self.entity_set and str3 not in self.stop_word_set:
                        # LogInfo.logs("Found 3-term pl [%s|%s|%s]",
                        #              spt[i], spt[i+1], spt[i+2])
                        new_line += "[[" + str3 + "]] "
                        i += 3
                        continue
                if i + 1 < len(spt):
                    str2 = spt[i] + spt[i + 1]
                    if str2 in self.entity_set and str2 not in self.stop_word_set:
                        # LogInfo.logs("Found 2-term pl [%s|%s]",
                        #              spt[i], spt[i+1])
                        new_line += "[[" + str2 + "]] "
                        i += 2
                        continue
                if spt[i] in self.entity_set and spt[i] not in self.stop_word_set:
                    # LogInfo.logs("Found pl [%s]", spt[i])
                    new_line += "[[" + spt[i] + "]] "
                    i += 1
                    continue
                new_line += spt[i] + " "
                i += 1
            fout.write(new_line + "\n")
            cnt += 1
            if cnt < 10:
                LogInfo.logs("res ==> (%s)", new_line)
            LogInfo.show_line(cnt, 100000)
        fin.close()
        fout.close()
        LogInfo.end_track("Entity tags added.")


class DataAdapter(object):
    """
    raw_data ---> model data
    """
    def __init__(self):
        self.root_fp = "/u01/xusheng/data"

    def prepare_model_data(self):
        LogInfo.begin_track("Generating model data...")
        # query_parse.rule: Q\tPL\tPK\tPV\n
        # PL/PK/PV : 1,2 3,4     from 1 not 0
        vocab = EntityAdder()
        vocab_set = dict()
        for string in vocab.entity_set:
            vocab_set[string] = set()
            for ch in string:
                vocab_set[string].add(ch)
        LogInfo.logs("Vocab char set processed. %d", len(vocab_set))
        fin = codecs.open(self.root_fp + "/query_parse.rule",
                          'r', encoding='utf-8')
        fout = codecs.open(self.root_fp + "/NLU_model_data_train.name",
                           'w', encoding='utf-8')
        cnt = 0
        for line in fin:
            cnt += 1
            if cnt % 10 == 0:
                LogInfo.logs("%d lines processed.", cnt)
                fout.flush()
            query, pl, pk, pv = line.split("\t")
            ret = ""
            q_len = len(query.split(" "))
            # query
            ret += query + "\t"
            # label & link_mask & entity name
            label = ["0"] * q_len
            link_mask = list()
            ents = list()
            if pl != "":
                pl_spt = pl.split(" ")
                for ij in pl_spt:
                    spt = ij.split(",")
                    i = int(spt[1])
                    j = int(spt[2])
                    name = spt[0]
                    for k in range(i, j+1):
                        label[k] = "2"  # PL_I
                        link_tmp = ["0"] * q_len
                        link_tmp[k] = "1"
                        link_mask.append(link_tmp)
                        ents.append(name)
                    label[i] = "1"  # PL_B
            if pk != "":
                pk_spt = pk.split(" ")
                for ij in pk_spt:
                    spt = ij.split(",")
                    i = int(spt[1])
                    j = int(spt[2])
                    name = spt[0]
                    for k in range(i, j+1):
                        label[k] = "4"  # PK_I
                        link_tmp = ["0"] * q_len
                        link_tmp[k] = "1"
                        link_mask.append(link_tmp)
                        ents.append(name)
                    label[i] = "3"  # PK_B
            if pv != "":
                pv_spt = pv.split(" ")
                for ij in pv_spt:
                    spt = ij.split(",")
                    i = int(spt[1])
                    j = int(spt[2])
                    name = spt[0]
                    for k in range(i, j+1):
                        label[k] = "6"  # PV_I
                        link_tmp = ["0"] * q_len
                        link_tmp[k] = "1"
                        link_mask.append(link_tmp)
                        ents.append(name)
                    label[i] = "5"  # PV_B
            # label
            ret += " ".join(label)
            ret += "\t"
            # intent, currently no intent data
            ret += "0\t"  # 0: PL, 1: PK, 2: PV
            # link_mask & entity_idx
            for _link_mask, _ent in zip(link_mask, ents):
                fout.write(ret + " ".join(_link_mask) + "\t" + _ent + " ")
                cands = fuzzy_match_name(_ent, vocab_set, 20)
                fout.write(" ".join(cands) + "\n")
        fin.close()
        fout.close()
        LogInfo.end_track()

if __name__ == "__main__":
    # worker = EntityAdder()
    # worker.add_entity_tag_yyh()
    runner = DataAdapter()
    runner.prepare_model_data()
