from math import log, sqrt
from prior import Prior
from wikigraph import WikiGraph


class Features:
    def __init__(self, prior, graph, context, weight, all_entities):
        self.prior = prior
        self.graph = graph
        self.entity_context = context
        self.weight = weight
        self.all_entities = all_entities
    """
    def feat_prior(self, mention):
        feats = []
        # 1. case-sensitive
        pairs = self.prior.prior[mention.surface]
        for e, p in pairs:
            if e == mention.entity:
                feats.append(p)
                break
        # 2. case-insensitive
        pairs = self.prior.prior_lower[mention.surface.lower()]
        for e, p in pairs:
            if e == mention.entity:
                feats.append(p)
                break
        
        return tuple(feats)
    """
    def feat_prior(self, mention):
        return tuple(mention.foo)

    def feat_semantic_relatedness(self, entity, pos, table):
        ## coherence between entities
        # Milne-Witten measure
        def sr_in_score(e1, e2):
            A = self.graph.get_in_link(e1)
            B = self.graph.get_in_link(e2)
            a, b, c = max(len(A), len(B)), len(A & B), min(len(A), len(B))
            if a == 0 or b == 0 or c == 0:
                return 1
            score = (log(a) - log(b)) / (log(self.graph.size) - log(c))
            return score
        def sr_out_score(e1, e2):
            A = self.graph.get_out_link(e1)
            B = self.graph.get_out_link(e2)
            a, b, c = max(len(A), len(B)), len(A & B), min(len(A), len(B))
            if a == 0 or b == 0 or c == 0:
                return 1
            score = (log(a) - log(b)) / (log(self.graph.size) - log(c))
            return score
               
        # sum(list, []) means to flatten the 2-dim list
        # row_in_score = sum([sr_in_score(entity, mention.entity) for mention in cell.mentions for cell in table.get_row_context(pos)], [])
        # row_out_score = sum([sr_out_score(entity, mention.entity) for mention in cell.mentions for cell in table.get_row_context(pos)], [])
        # col_in_score = sum([sr_in_score(entity, mention.entity) for mention in cell.mentions for cell in table.get_col_context(pos)], [])
        # col_out_score = sum([sr_out_score(entity, mention.entity) for mention in cell.mentions for cell in table.get_col_context(pos)], [])
        row_in_score = [sr_in_score(entity, cell.label) for cell in table.get_row_context(pos)]
        row_out_score = [sr_out_score(entity, cell.label) for cell in table.get_row_context(pos)]
        col_in_score = [sr_in_score(entity, cell.label) for cell in table.get_col_context(pos)]
        col_out_score = [sr_out_score(entity, cell.label) for cell in table.get_col_context(pos)]
        if len(row_in_score) == 0:
            score1, score2 = 1, 1
        else:
            score1, score2 = sum(row_in_score) / len(row_in_score), sum(row_out_score) / len(row_out_score)
        if len(col_in_score) == 0:
            score3, score4 = 1, 1
        else:
            score3, score4 = sum(col_in_score) / len(col_in_score), sum(col_out_score) / len(col_out_score)
        if len(row_in_score) + len(col_in_score) == 0:
            score5, score6 = 1, 1
        else:
            score5, score6 = sum(row_in_score + col_in_score) / (len(row_in_score) + len(col_in_score)), sum(row_out_score + col_out_score) / (len(row_out_score) + len(col_out_score))
        features = (score1, score2, score3, score4, score5, score6)
        return features

    def feat_mention_entity_similarity(self, table, pos, entity):
        ## entity context: pre-computed, just look up table
        if entity in self.entity_context[0] and entity in self.entity_context[1]:
            entity_context_word, entity_cotext_entity = self.entity_context[0][entity], self.entity_context[1][entity] #
        else:
            return (0, 0, 0, 0, 0, 0)
        
        ## mention context
        # row context
        row_context = table.get_row_context(pos)
        row_word = dict()
        row_entity = dict()
        for cell in row_context:
            tokens = cell.surface.split(' ')
            for token in tokens:
                if token in row_word:
                    row_word[token] = row_word[token] + 1
                else:
                    row_word[token] = 1
            if cell.label:
                if cell.label in row_entity:
                    row_entity[cell.label] = row_entity[cell.label] + 1
                else:
                    row_entity[cell.label] = 1
        # column context
        col_context = table.get_col_context(pos)
        col_word = dict()
        col_entity = dict()
        for cell in col_context:
            tokens = cell.surface.split(' ')
            for token in tokens:
                if token in col_word:
                    col_word[token] = col_word[token] + 1
                else:
                    col_word[token] = 1
            if cell.label:
                if cell.label in col_entity:
                    col_entity[cell.label] = col_entity[cell.label] + 1
                else:
                    col_entity[cell.label] = 1
        # all context
        all_word = merge_two_dicts(col_word, row_word)
        all_entity = merge_two_dicts(col_entity, row_entity)

        def cos(ms1, ms2, type):
            if len(ms1) == 0 or len(ms2) == 0:
                return 0
            # print '1:', ms1.keys()
            # print '2:', ms2.keys()
            # weight (don't use now)
            # """
            # if type == 'w':
            #     for word, frequency in ms1.items():
            #         weighted_frequency = frequency * self.weight.word[word]
            #         ms1[word] = weighted_frequency
            #     for word, frequency in ms2.items():
            #         weighted_frequency = frequency * self.weight.word[word]
            #         ms2[word] = weighted_frequency
            # elif type == 'e':
            #     for entity, frequency in ms1.items():
            #         weighted_frequency = frequency * self.weight.entity[entity]
            #         ms1[entity] = weighted_frequency
            #     for entity, frequency in ms2.items():
            #         weighted_frequency = frequency * self.weight.entity[entity]
            #         ms2[entity] = weighted_frequency
            # else:
            #     raise
            # """
            # cosine similarity
            shared = set(ms1.keys()).intersection(set(ms2.keys()))
            # print '3:', shared
            dotproduct = sum([ms1[key] * ms2[key] for key in shared])
            module = sqrt(sum([f * f for f in ms1.values()]) * sum([f * f for f in ms2.values()]))
            sim = dotproduct / module
            return sim

        features = cos(row_word, entity_context_word, 'w'), cos(col_word, entity_context_word, 'w'), cos(all_word, entity_context_word, 'w'), \
            cos(row_entity, entity_cotext_entity, 'e'), cos(col_entity, entity_cotext_entity, 'e'), cos(all_entity, entity_cotext_entity, 'e')
        return features


    def feat_existing_link(self, table, mention, entity, pos):
        # 1. whether the mention's context contains another identical surface form which is also linked to the candidate
        feat1 = 0
        context = table.get_context(pos)
        for cell in context:
            for cmention in cell.mentions:
                if cmention.surface == mention.surface and cmention.entity == entity:
                    feat1 = 1
                    break
        # 2. whether there is a different surface form which is also linked to the candidate
        feat2  = 0
        for cell in table.cells:
            for cmention in cell.mentions:
                if cmention.entity == entity and cmention.surface != mention.surface:
                    feat2 = 1
                    break
        return feat1, feat2

    def feat_surface(self, mention):
        # 1. the mention's surface is exactly the cell's surface
        feat1 = 0
        if mention.surface == mention.cell.surface:
            feat1 = 1
        # 2. the mention's surface exactly matches an entity in kb
        feat2 = 0
        if mention.surface in self.all_entities:
            feat2 = 1

        return feat1, feat2

    def extract_features(self, table, pos, mention, entity):
        return tuple(self.feat_prior(mention) + self.feat_semantic_relatedness(entity, pos, table) + self.feat_mention_entity_similarity(table, pos, entity)) \
            + self.feat_existing_link(table, mention, entity, pos) + self.feat_surface(mention)

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z