from entity_linker import entity_linker, surface_index_memory
from corenlp_parser import parser
import logging

class EntityLinker:
    def __init__(self):
        base = '/home/xianyang/aqqu/aqqu'
        print('initiating parser...')
        self.parser = parser.CoreNLPParser('http://202.120.38.146:8666/parse')
        print('initiating index...')
        self.surface_index = surface_index_memory.EntitySurfaceIndexMemory \
            (base + '/data/entity-list', base + '/data/entity-surface-map', base + '/data/entity-index')
        print('initiating entity linker...')
        self.entity_linker = entity_linker.EntityLinker(self.surface_index, 7)
        print('initiation done.')

    def link(self, question):
        parse_result = self.parser.parse(question)
        identified = self.entity_linker.identify_entities_in_tokens(parse_result.tokens)
        # return [(e.entity.id, e.score) for e in identified]
        return identified