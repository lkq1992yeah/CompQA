# -*- code: utf-8 -*-
from table import *
from feature import *
import json
import codecs

def candidate_generation(table, prior):
    for cell in table.cells:
        if cell.blank:
            continue
        surface = cell.surface
        surface_lower = surface.lower()
        if surface_lower in prior.prior_lower:
            for e, prob in prior.prior_lower[surface_lower]:
                if prob > 0:
                    cell.mentions.append(Mention(surface_lower, e, cell, cell.pos))

def candidate_generation_from_oracle():
    pass


def feature_extraction(table, features):
    for cell in table.cells:
        for mention in cell.mentions:
            feat = features.extract_features(table, cell.pos, mention, mention.entity)

def load_entity_context():
    entity_context_word = dict()
    with open('entity_context_word.txt', 'r') as f:
        for line in f:
            e_and_context = line[:-1].split('\t\t\t\t')
            e, context = e_and_context[0], e_and_context[1]
            entity_context_word[e] = dict()
            context = context.split('\t\t')
            for word_and_frequency in context:
                tmp = word_and_frequency.split('\t')
                word, frequency = tmp[0], tmp[1]
                entity_context_word[e][word] = float(frequency)
    entity_context_entity = dict()
    with open('entity_context_entity.txt', 'r') as f:
        for line in f:
            e_and_context = line[:-1].split('\t\t\t\t')
            e, context = e_and_context[0], e_and_context[1]
            entity_context_entity[e] = dict()
            context = context.split('\t\t')
            for entity_and_frequency in context:
                tmp = entity_and_frequency.split('\t')
                entity, frequency = tmp[0], tmp[1]
                entity_context_entity[e][entity] = float(frequency)

    return entity_context_word, entity_context_entity

def preprocess():
    print('loading prior...')
    prior = Prior('/home/xusheng/wikipedia/zh-extracted/prior.txt')
    print('loading wikipedia link graph...')
    wikigraph = WikiGraph('wiki_link_graph.txt')
    print('loading entity context...')
    entity_context = load_entity_context()
    weight = None
    # all_entities = set(prior.prior.keys()) # fix
    all_entities = set()
    with open('/home/xusheng/wikipedia/zh-extracted/entity.txt', 'r') as f:
        for line in f:
            all_entities.add(line[:-1])

    return prior, wikigraph, entity_context, weight, all_entities

def load_dict():
    zh2en = {}
    zh2cand = {}
    with open('/home/xusheng/PythonProject/data/tabel/size_116/translation.txt.116', 'r') as f:
        for line in f:
            splited = line[:-1].split('\t')
            zh = splited[0].decode('utf8')
            en = splited[3]
            zh2en[zh] = en
    with codecs.open('/home/xusheng/PythonProject/data/tabel/monolingual/candCell_50.txt.all.detail', 'r',
                     encoding='utf-8') as f:
        for line in f:
            splited = line[:-1].split('\t')
            zh = splited[0]
            cands = []
            for i in range(1, len(splited)):
                entity, score = splited[i].split('|||')
                entity = entity[2:-2].replace('_', ' ')
                score = float(score)
                cands.append((entity, score))
            zh2cand[zh] = cands

    return zh2en, zh2cand