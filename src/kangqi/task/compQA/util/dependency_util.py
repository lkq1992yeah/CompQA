import os
import json

from ..linking.parser import CoreNLPParser, parse_json_result

from kangqi.util.LogUtil import LogInfo


type_replace_tups = [
    ('people.person', 'Richardson'),
    ('location.location', 'California'),
    ('organization.organization', 'Microsoft'),
    ('sports.sports_team', 'Yankees'),
    ('film.film', 'Titanic'),
    ('film.film_character', 'Alice'),
    ('tv.tv_program', 'Titanic'),
    ('tv.tv_character', 'Alice'),
    ('book.book', 'Hobbits'),
    ('music.album', 'Hobbits'),
    ('music.composition', 'Hobbits'),
    ('music.musical_group', 'Gala'),
    ('time.event', 'WWII'),
    ('people.profession', 'lawyer'),
]


class DependencyUtil:

    def __init__(self, e_t_dict,
                 dep_cache_fp='runnings/acl18_cache/dep_parse.txt',
                 # path_cache_fp='runnings/acl18_cache/find_path.txt',
                 parser_ip='202.120.38.146', parser_port=9601):
        self.e_t_dict = e_t_dict
        self.parser = CoreNLPParser('http://%s:%d/parse' % (parser_ip, parser_port))  # just open the parser
        self.dep_cache_fp = dep_cache_fp
        # self.path_cache_fp = path_cache_fp
        self.dep_cache_dict = {}
        self.path_cache_dict = {}
        self.verbose = 0
        LogInfo.begin_track('DependencyUtil reading cache ...')
        if os.path.isfile(self.dep_cache_fp):
            with open(self.dep_cache_fp, 'r') as br:
                for line_idx, line in enumerate(br.readlines()):
                    # LogInfo.logs('Reading dependency parsing: line_idx = %d ...', line_idx+1)
                    utterance, json_response = json.loads(line)
                    parse_result = parse_json_result(json_response=json_response.encode('utf-8'),
                                                     include_json=True, parse_trees=True)
                    self.dep_cache_dict[utterance] = parse_result.dependency_parse
        LogInfo.logs('%d dependency cache loaded.', len(self.dep_cache_dict))
        # if os.path.isfile(self.path_cache_fp):
        #     with open(self.path_cache_fp, 'r') as br:
        #         for line in br.readlines():
        #             k, v = line.strip().split('|||')
        #             path_tok_list = v.split('\t')
        #             self.path_cache_dict[k] = path_tok_list
        # LogInfo.logs('%d target path cache loaded.', len(self.path_cache_dict))
        LogInfo.end_track()

    def context_pattern(self, tok_list, linkings):
        """
        Imitate Bao's context pattern
        :param tok_list:
        :param linkings:
        :return:
        """
        ph_tok_list, link_anchor_list, ans_anchor = self.placeholding(tok_list=tok_list, linkings=linkings)
        for link_anchor, gl_data in zip(link_anchor_list, linkings):
            if gl_data.category == 'Entity':
                ph_tok_list[link_anchor] = '<E>'
            elif gl_data.category == 'Time':
                ph_tok_list[link_anchor] = '<Tm>'       # immediately replace

        window = 2
        ph_tok_len = len(ph_tok_list)
        cp_tok_lists = []
        for link_anchor in link_anchor_list:
            st_pos = max(0, link_anchor-window)
            ed_pos = min(ph_tok_len-1, link_anchor+window) + 1
            cp_toks = ph_tok_list[st_pos: ed_pos]
            cp_tok_lists.append(cp_toks)
        return cp_tok_lists

    def dep_path_seq(self, tok_list, linkings):
        if self.verbose >= 1:
            LogInfo.begin_track('Work for dependency paths:')
            LogInfo.logs('Original token: %s', tok_list)
            for gl_data in linkings:
                LogInfo.logs('Linking: [%s] [%d, %d)', gl_data.category, gl_data.start, gl_data.end)

        ph_tok_list, link_anchor_list, ans_anchor = self.placeholding(tok_list=tok_list, linkings=linkings)
        if self.verbose >= 1:
            LogInfo.logs('Placeholder token: %s', ph_tok_list)
            LogInfo.logs('link_anchor_list: %s', link_anchor_list)
            LogInfo.logs('ans_anchor: %d', ans_anchor)
        placeholder_dict = {}
        for link_anchor, gl_data in zip(link_anchor_list, linkings):
            if gl_data.category == 'Entity':
                placeholder_dict[link_anchor] = '<E>'
            elif gl_data.category == 'Time':
                placeholder_dict[link_anchor] = '<Tm>'

        utterance = ' '.join(ph_tok_list)
        if utterance not in self.dep_cache_dict:
            parse_result = self.parser.parse(utterance, parse_trees=True, include_json=True)
            dependency_parse = parse_result.dependency_parse
            self.dep_cache_dict[utterance] = dependency_parse
            with open(self.dep_cache_fp, 'a') as bw:
                tup = (utterance.encode('utf-8'), parse_result.json_result.encode('utf-8'))
                bw.write(json.dumps(tup) + '\n')
        else:
            dependency_parse = self.dep_cache_dict[utterance]

        edge_dict = {}
        for rel, head, dep in dependency_parse.dependencies:
            head_pos = int(head[head.rindex('-')+1:])
            dep_pos = int(dep[dep.rindex('-')+1:])
            fwd_key = '%d-%d' % (head_pos, dep_pos)
            bkwd_key = '%d-%d' % (dep_pos, head_pos)
            edge_dict[fwd_key] = rel
            edge_dict[bkwd_key] = '!%s' % rel
        if self.verbose >= 1:
            LogInfo.logs('Edge_dict: %s', edge_dict)

        path_tok_lists = []
        for link_idx, link_anchor in enumerate(link_anchor_list):
            category = linkings[link_idx].category
            if self.verbose >= 1:
                LogInfo.begin_track('path for linking %d / %d:', link_idx, len(link_anchor_list))
            # key = '%s|%d|%d|%s' % (utterance, link_anchor, ans_anchor, category)
            # if key not in self.path_cache_dict:
            path_tok_list = self.find_path(dep_parse=dependency_parse,
                                           link_anchor=link_anchor,
                                           ans_anchor=ans_anchor,
                                           link_category=category,
                                           edge_dict=edge_dict,
                                           ph_dict=placeholder_dict)
            #     self.path_cache_dict[key] = path_tok_list
            #     with open(self.path_cache_fp, 'a') as bw:
            #         bw.write('%s|||%s\n' % (key.encode('utf-8'),
            #                                 '\t'.join(path_tok_list).encode('utf-8')))
            # else:
            #     path_tok_list = self.path_cache_dict[key]
            path_tok_lists.append(path_tok_list)
            if self.verbose >= 1:
                LogInfo.end_track()

        if self.verbose >= 1:
            LogInfo.end_track()
        return path_tok_lists

    def placeholding(self, tok_list, linkings):
        """ Given the original tokens, replace detail linking data by specific words or tokens """

        """ Step 1: Shrink each E/Tm link, occupying only one token """
        ph_tok_list = list(tok_list)
        link_pos_list = [-1] * len(tok_list)
        for link_idx, gl_data in enumerate(linkings):
            st = gl_data.start
            ed = gl_data.end
            link_pos_list[ed-1] = link_idx      # identifying the anchor word of the current linking
            if gl_data.category in ('Entity', 'Time'):  # only perform placeholder for E/Tm
                for tok_idx in range(st, ed-1):
                    ph_tok_list[tok_idx] = ''
        tok_link_tups = []
        for ph_tok, link_idx in zip(ph_tok_list, link_pos_list):
            if ph_tok != '':
                tok_link_tups.append([ph_tok, link_idx])        # remove non-trailing words of E/Tm

        link_anchor_list = [-1] * len(linkings)  # the anchor position
        for anchor_idx, tup in enumerate(tok_link_tups):
            link_idx = tup[-1]
            if link_idx != -1:
                link_anchor_list[link_idx] = anchor_idx

        """ Step 2: Determine answer's anchor point """
        ans_anchor = 0        # find the first wh- word in the sentence, otherwise picking the first word
        for tok_idx, (ph_tok, link_idx) in enumerate(tok_link_tups):
            if ph_tok.startswith('wh') or ph_tok == 'how':
                ans_anchor = tok_idx
                break

        """ Step 3: Dynamic replacement """
        for anchor_idx in range(len(tok_link_tups)):
            ph_tok, link_idx = tok_link_tups[anchor_idx]
            if link_idx == -1:
                continue
            gl_data = linkings[link_idx]
            cate = gl_data.category
            if cate in ('Type', 'Ordinal'):
                continue
            elif cate == 'Time':
                tok_link_tups[anchor_idx][0] = '2018'        # fix time
            else:
                entity_rep = 'XYZ'      # default value
                mid = gl_data.value
                mid_tp_set = self.e_t_dict.get(mid, set([]))
                for tp, rep in type_replace_tups:
                    if tp in mid_tp_set:
                        entity_rep = rep
                        break
                tok_link_tups[anchor_idx][0] = entity_rep
        final_ph_tok_list = [tup[0] for tup in tok_link_tups]

        return final_ph_tok_list, link_anchor_list, ans_anchor

    def find_path(self, dep_parse, link_anchor, ans_anchor, link_category, edge_dict, ph_dict):
        """
        :param dep_parse: dependency graph
        :param link_anchor: token index of the focus word (0-based)
        :param ans_anchor: token index of the answer (0-based)
        :param link_category: the category of the current focus link
        :param edge_dict: <head-dep, rel> dict
        :param ph_dict: <token_idx, ph> dict
        :return:
        """
        link_anchor += 1             # ROOT is at position 0
        ans_anchor += 1
        src_node = dst_node = None
        for node in dep_parse.graph.nodes:
            if node.position == ans_anchor:
                src_node = node
            if node.position == link_anchor:
                dst_node = node
        if self.verbose >= 1:
            LogInfo.logs('src_node: %s', src_node.word)
            LogInfo.logs('dst_node: %s', dst_node.word)
        if ans_anchor != link_anchor:
            path_nodes = dep_parse.graph.shortest_path(node_a=src_node, node_b=dst_node)
        else:       # just the node itself
            path_nodes = [dst_node]
        if self.verbose >= 1:
            LogInfo.logs('shortest path: %s', [p.word for p in path_nodes])

        path_tok_list = []
        path_len = len(path_nodes)
        if path_len > 0:
            for pos in range(path_len-1):
                key = '%d-%d' % (path_nodes[pos].position, path_nodes[pos+1].position)
                edge = edge_dict[key]
                cur_token_idx = path_nodes[pos].position - 1
                if cur_token_idx in ph_dict:    # replace other focus in the path
                    path_tok_list.append(ph_dict[cur_token_idx])
                else:
                    path_tok_list.append(path_nodes[pos].word)
                path_tok_list.append(edge)
            if link_category == 'Entity':
                path_tok_list.append('<E>')
            elif link_category == 'Time':
                path_tok_list.append('<Tm>')
            else:
                path_tok_list.append(path_nodes[-1].word)
        if self.verbose >= 1:
            LogInfo.logs('return path_tok_list: %s', path_tok_list)

        return path_tok_list
