


class Prior:
    def __init__(self, path):
        # dict: surface -> [(e, prob)]
        self.prior = dict()
        self.prior_lower = dict()
        with open(path, 'r') as f:
            for line in f:
                tmp = line[:-1].split('\t\t')
                mention = tmp[0]
                mention_lower = mention.lower()
                pairs = [] # [(mention_lower, 1.0)]
                for i in range(1, len(tmp)):
                    entity, prob = tmp[i].split('\t')
                    if not prob > 0:
                        break
                    pairs.append((entity, float(prob)))
                self.prior[mention] = pairs
                self.prior_lower[mention_lower] = pairs
