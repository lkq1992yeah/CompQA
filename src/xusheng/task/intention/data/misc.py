"""
misc utils for data processing & analysis & visualization
"""

from xusheng.util.log_util import LogInfo

import codecs


class PinleiAdder(object):

    def __init__(self):
        self.root_fp = "/u01/xusheng/data"
        self.pinlei_set = set()

    def load_pinlei(self):
        LogInfo.begin_track("Load pinlei names...")
        with codecs.open(self.root_fp + "/raw/kg_pinlei_id", 'r',
                         encoding='utf-8') as fin:
            for line in fin:
                spt = line.strip().split("\t")
                if len(spt) < 2:
                    continue
                pinlei = spt[0]
                self.pinlei_set.add(pinlei)
        LogInfo.end_track("%d names loaded.", len(self.pinlei_set))

    def add_pinlei_tag_yyh(self):
        LogInfo.begin_track("Begin adding tags for pinleis...")
        fin = codecs.open(self.root_fp + "/yyh_w2v_train.txt",
                          'r',
                          encoding='utf-8')
        fout = codecs.open(self.root_fp + "/yyh_w2v_train.txt.pinlei_tag",
                           'w',
                           encoding='utf-8')
        cnt = 0
        for line in fin:
            spt = line.strip().split()
            new_line = ""
            i = 0
            while i < len(spt):
                if i+3 < len(spt):
                    str4 = spt[i] + spt[i+1] + spt[i+2] + spt[3]
                    if str4 in self.pinlei_set:
                        LogInfo.logs("Found 4-term pinlei [%s|%s|%s|%s]",
                                     spt[i], spt[i+1], spt[i+2], spt[i+3])
                        new_line += "[[" + str4 + "]] "
                        i += 4
                        continue
                if i+2 < len(spt):
                    str3 = spt[i] + spt[i+1] + spt[i+2]
                    if str3 in self.pinlei_set:
                        # LogInfo.logs("Found 3-term pinlei [%s|%s|%s]",
                        #              spt[i], spt[i+1], spt[i+2])
                        new_line += "[[" + str3 + "]] "
                        i += 3
                        continue
                if i+1 < len(spt):
                    str2 = spt[i] + spt[i+1]
                    if str2 in self.pinlei_set:
                        # LogInfo.logs("Found 2-term pinlei [%s|%s]",
                        #              spt[i], spt[i+1])
                        new_line += "[[" + str2 + "]] "
                        i += 2
                        continue
                if spt[i] in self.pinlei_set:
                    # LogInfo.logs("Found pinlei [%s]", spt[i])
                    new_line += "[[" + spt[i] + "]] "
                    i += 1
                    continue
                new_line += spt[i] + " "
                i += 1
            fout.write(new_line + "\n")
            cnt += 1
            if cnt < 5:
                LogInfo.logs("res ==> (%s)", new_line)
            LogInfo.show_line(cnt, 100000)
        fin.close()
        fout.close()
        LogInfo.end_track("Pinlei tags added.")

    def process_query(self):
        LogInfo.begin_track("Begin adding tags for queries...")
        fin = codecs.open(self.root_fp + "/query.txt",
                          'r',
                          encoding='utf-8')
        fout = codecs.open(self.root_fp + "/query_label.txt",
                           'w',
                           encoding='utf-8')
        cnt = 0
        for line in fin:
            spt = line.strip().split()
            new_line = ""
            context = ""
            label = set()
            i = 0
            while i < len(spt):
                if i+4 < len(spt):
                    str5 = spt[i] + spt[i+1] + spt[i+2] + spt[i+3] + spt[i+4]
                    if str5 in self.pinlei_set:
                        LogInfo.logs("Found 5-term pinlei [%s|%s|%s|%s|%s]",
                                     spt[i], spt[i+1], spt[i+2], spt[i+3], spt[i+4])
                        label.add(str5)
                        new_line += "[[" + str5 + "]] "
                        i += 5
                        continue
                if i+3 < len(spt):
                    str4 = spt[i] + spt[i+1] + spt[i+2] + spt[i+3]
                    if str4 in self.pinlei_set:
                        LogInfo.logs("Found 4-term pinlei [%s|%s|%s|%s]",
                                     spt[i], spt[i+1], spt[i+2], spt[i+3])
                        label.add(str4)
                        new_line += "[[" + str4 + "]] "
                        i += 4
                        continue
                if i+2 < len(spt):
                    str3 = spt[i] + spt[i+1] + spt[i+2]
                    if str3 in self.pinlei_set:
                        LogInfo.logs("Found 3-term pinlei [%s|%s|%s]",
                                     spt[i], spt[i+1], spt[i+2])
                        label.add(str3)
                        new_line += "[[" + str3 + "]] "
                        i += 3
                        continue
                if i+1 < len(spt):
                    str2 = spt[i] + spt[i+1]
                    if str2 in self.pinlei_set:
                        # LogInfo.logs("Found 2-term pinlei [%s|%s]",
                        #              spt[i], spt[i+1])
                        label.add(str2)
                        new_line += "[[" + str2 + "]] "
                        i += 2
                        continue
                if spt[i] in self.pinlei_set:
                    # LogInfo.logs("Found pinlei [%s]", spt[i])
                    label.add(spt[i])
                    new_line += "[[" + spt[i] + "]] "
                    i += 1
                    continue
                context += spt[i] + " "
                new_line += spt[i] + " "
                i += 1

            if len(label) != 0:
                ret = new_line.strip() + "\t" + \
                      context.strip() + "\t" + \
                      "\t".join(label) + "\n"
            else:
                ret = new_line.strip() + "\n"
            fout.write(ret)
            cnt += 1
            if cnt < 5:
                LogInfo.logs("res ==> (%s)", ret.strip())
            LogInfo.show_line(cnt, 100000)
        fin.close()
        fout.close()
        LogInfo.end_track("Query processed.")


class DataAdapter(object):
    """
    data adapter from formatted data to
    """
    def __init__(self):
        self.root_fp = "/u01/xusheng/data/pinlei"
        self.pinlei = set()

    def load_pinlei(self):
        LogInfo.begin_track("Load pinlei names...")
        with codecs.open("/u01/xusheng/word2vec/vec/yyh_pinlei.txt",
                         'r', encoding='utf-8') as fin:
            for line in fin:
                name = line.strip().split()[0]
                if name.startswith("[["):
                    self.pinlei.add(name)
        LogInfo.end_track("Pinlei name loaded. Size: %d.", len(self.pinlei))

    def neg_sample_random(self, pos, num):
        negs = set()
        source = list(self.pinlei)
        import random
        while len(negs) < num:
            idx = random.randint(0, len(source)-1)
            if source[idx] != pos:
                negs.add(source[idx])
        # LogInfo.logs("negs: %s", negs)
        return negs

    def prepare_model_data(self):
        LogInfo.begin_track("Generate model data...")
        # .1 means single pinlei
        fin = codecs.open(self.root_fp + "/query_label.txt.1",
                          'r', encoding='utf-8')
        fout = codecs.open(self.root_fp + "/model_data_train.name",
                           'w', encoding='utf-8')
        not_cover = 0
        not_context = 0
        cnt = 0
        for line in fin:
            cnt += 1
            if cnt % 100000 == 0:
                LogInfo.logs("%d lines processed.", cnt)
                fout.flush()
            spt = line.strip().split("\t")
            context = spt[1]
            pinlei = "[[" + spt[2] + "]]"
            if pinlei not in self.pinlei:
                not_cover += 1
                continue
            if len(spt[1].split(" ")) < 6 or len(spt[1].split(" ")) > 15:
                not_context += 1
                continue
            fout.write(context + "\t" + pinlei + "\n")
            negs = self.neg_sample_random(pinlei, 19)
            for neg in negs:
                fout.write(context + "\t" + neg + "\n")

        fin.close()
        fout.close()
        LogInfo.end_track("Model data prepared. Size: %d. (%d, %d).",
                          cnt-not_context-not_cover, not_cover, not_context)


class MultiPinleiEvalDataAdapter(DataAdapter):

    def __init__(self):
        super(MultiPinleiEvalDataAdapter, self).__init__()
        self.pinlei_num = 0

    def prepare_model_data(self, pinlei_num):
        self.pinlei_num = pinlei_num
        LogInfo.begin_track("Generate Multi-Pinlei Data for evaluation...")
        fin = codecs.open(self.root_fp + "/query_label.txt." + str(self.pinlei_num),
                          'r', encoding='utf-8')
        fout = codecs.open(self.root_fp + "/model_data_test." +
                           str(self.pinlei_num) + ".name",
                           'w', encoding='utf-8')
        fsho = codecs.open(self.root_fp + "/model_data_test." +
                           str(self.pinlei_num) + ".check",
                           'w', encoding='utf-8')
        cnt = 0
        not_cover = set()
        for line in fin:
            cnt += 1
            if cnt % 100000 == 0:
                LogInfo.logs("%d lines processed.", cnt)
                fout.flush()
            spt = line.strip().split("\t")
            context = spt[1]
            is_cover = True
            for i in range(2, 2+self.pinlei_num):
                pinlei = "[[" + spt[i] + "]]"
                if pinlei not in self.pinlei:
                    # LogInfo.logs("%s not cover.", pinlei)
                    is_cover = False
                    not_cover.add(pinlei)
            if not is_cover:
                continue
            if len(spt[1].split(" ")) < 6 or len(spt[1].split(" ")) > 15:
                continue
            for i in range(2, 2+self.pinlei_num):
                pinlei = "[[" + spt[i] + "]]"
                fout.write(context + "\t" + pinlei + "\n")
                fsho.write(spt[0] + "\n")

        fin.close()
        fout.close()
        fsho.close()
        LogInfo.end_track("%d pinlei not cover.", len(not_cover))

    def tag_pinlei(self, query):
        LogInfo.logs("Tagging pinlei for your query...")
        spt = query.strip().split()
        new_line = ""
        context = ""
        label = set()
        i = 0
        while i < len(spt):
            if i+4 < len(spt):
                str5 = spt[i] + spt[i+1] + spt[i+2] + spt[i+3] + spt[i+4]
                if "[[" + str5 + "]]" in self.pinlei:
                    LogInfo.logs("Found 5-term pinlei [%s|%s|%s|%s|%s]",
                                 spt[i], spt[i+1], spt[i+2], spt[i+3], spt[i+4])
                    label.add("[[" + str5 + "]]")
                    new_line += "[[" + str5 + "]] "
                    i += 5
                    continue
            if i+3 < len(spt):
                str4 = spt[i] + spt[i+1] + spt[i+2] + spt[i+3]
                if "[[" + str4 + "]]" in self.pinlei:
                    LogInfo.logs("Found 4-term pinlei [%s|%s|%s|%s]",
                                 spt[i], spt[i+1], spt[i+2], spt[i+3])
                    label.add("[[" + str4 + "]]")
                    new_line += "[[" + str4 + "]] "
                    i += 4
                    continue
            if i+2 < len(spt):
                str3 = spt[i] + spt[i+1] + spt[i+2]
                if "[[" + str3 + "]]" in self.pinlei:
                    LogInfo.logs("Found 3-term pinlei [%s|%s|%s]",
                                 spt[i], spt[i+1], spt[i+2])
                    label.add("[[" + str3 + "]]")
                    new_line += "[[" + str3 + "]] "
                    i += 3
                    continue
            if i+1 < len(spt):
                str2 = spt[i] + spt[i+1]
                if "[[" + str2 + "]]" in self.pinlei:
                    # LogInfo.logs("Found 2-term pinlei [%s|%s]",
                    #              spt[i], spt[i+1])
                    label.add("[[" + str2 + "]]")
                    new_line += "[[" + str2 + "]] "
                    i += 2
                    continue
            if "[[" + spt[i] + "]]" in self.pinlei:
                # LogInfo.logs("Found pinlei [%s]", spt[i])
                label.add("[[" + spt[i] + "]]")
                new_line += "[[" + spt[i] + "]] "
                i += 1
                continue
            context += spt[i] + " "
            new_line += spt[i] + " "
            i += 1

        return new_line.strip(), context.strip(), list(label)


if __name__ == "__main__":
    worker = PinleiAdder()
    worker.load_pinlei()
    worker.add_pinlei_tag_yyh()
    # worker.process_query()

    # adapter = DataAdapter()
    # adapter.load_pinlei()
    # adapter.prepare_model_data()

    # import sys
    # adapter = MultiPinleiEvalDataAdapter()
    # adapter.load_pinlei()
    # adapter.prepare_model_data(int(sys.argv[1]))


