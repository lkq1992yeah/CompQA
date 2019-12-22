# We redefine the Schema class, for ICDE 2018 rush.
class Schema(object):

    def __init__(self, path_list, comb=None, path_len=1, original_idx=-1, use_idx=-1, p=0., r=0., f1=0.):
        """
        :param path_list: a list of paths, each path is a list of predicates
        :param comb: the combination detail, indicating which entity links to which node in the skeleton
        :param path_len: the length of the main skeleton (no use in our model)
        :param original_idx: (deprecated) indicating the original schema index before any filtering
        :param use_idx: the schema index after possible filtering steps
        :param p: precision
        :param r: recall
        :param f1: f1
        """
        self.path_list = path_list
        self.path_len = path_len            # no use now
        self.comb = comb
        self.path_words_list = None         # show the words of each path

        self.original_idx = original_idx    # the index of the schema
        self.use_idx = use_idx

        self.p = p
        self.r = r
        self.f1 = f1

        self.run_info = None

    def disp(self):
        names = ['-->'.join(path) for path in self.path_list]
        names.sort()
        return '\t'.join(names)

    def is_schema_ok(self, sc_max_len, path_max_len):
        if len(self.path_list) > sc_max_len:    # sc_max_len exceeds the limit
            return False
        len_exceed = False
        for path in self.path_list:
            if len(path) - 1 > path_max_len:    # path length exceeds the limit
                len_exceed = True
                break
        if len_exceed:
            return False
        return True

    def update_item_set(self, e_set, t_set, p_set):
        for path in self.path_list:
            if len(path) == 1:      # SimpQ scenario
                pred = path[0]
                p_set.add(pred)
            else:                   # WebQ / CompQ
                # TODO: is focus at the beginning, or the end?? Be careful.
                focus = path[0]
                if focus.startswith('m.') or focus.find('.') == -1:
                    # common entities or numeric/time/ordinal value
                    e_set.add(focus)
                else:
                    t_set.add(focus)
                for pred in path[1:]:
                    p_set.add(pred)     # both forward and backward predicates could be kept


def is_intermediate_node(node_name):
    if node_name == 'x' or (node_name.startswith('y') and len(node_name) == 2):     # x, y0, y1, ...
        return True
    return False


# DFS starting from x, and revert all edges
def build_sc_from_line(schema_line, fb_helper):
    # LogInfo.logs('sc_line: [%s]', schema_line)
    # Step 1: create adjacent matrix
    buf = schema_line.split('\t')
    edge_dict = {}
    non_dup = 0         # append to entity / ordinal / time information, avoid duplicate entities in the schema
    for line in buf:
        subj, pred, obj = line.split('--')
        if not is_intermediate_node(subj):      # if the node is intermediate, then no need to add the non_dup mark
            subj = subj + '###%d' % non_dup
            non_dup += 1
        if not is_intermediate_node(obj):
            obj = obj + '###%d' % non_dup
            non_dup += 1
        if subj not in edge_dict:
            edge_dict[subj] = []
        if obj not in edge_dict:
            edge_dict[obj] = []
        edge_dict[subj].append((pred, obj))
        edge_dict[obj].append((fb_helper.inverse_predicate(pred), subj))
    # Step 2: DFS from x
    visit_set = {'x'}
    cur_node = 'x'
    cur_path = []
    path_list = []
    dfs_search(fb_helper, edge_dict, visit_set, cur_path, cur_node, path_list)
    sc = Schema(path_list=path_list)
    path_list_str = sc.disp()
    return sc, path_list_str


def dfs_search(fb_helper, edge_dict, visit_set, cur_path, cur_node, path_list):
    flag = False
    for pred, target in edge_dict[cur_node]:
        if target in visit_set:     # already visited
            continue
        flag = True
        cur_path = [fb_helper.inverse_predicate(pred)] + cur_path       # revert edge, put at the front
        visit_set.add(target)
        dfs_search(fb_helper, edge_dict, visit_set, cur_path, target, path_list)
        cur_path = cur_path[1:]
        visit_set.remove(target)
    if not flag:            # no way out, hit the leaf
        leaf_node = cur_node.split('###')[0]
        new_path = [leaf_node] + list(cur_path)
        path_list.append(new_path)


# deprecated due to bugs
# def build_sc_from_line(schema_line, fb_helper):
#     LogInfo.logs(schema_line)
#     # Step 1: Extract and prepare all subj --> (pred, obj) information
#     buf = schema_line.split('\t')
#     path_tup_list = []          # [(path, path_str)]
#     edge_dict = {}
#     for line in buf:
#         subj, pred, obj = line.split('--')  # subj, pred, obj
#         # Now checking whether to change the order
#         reverse = False
#         if not is_intermediate_node(obj):
#             reverse = True
#         elif subj == 'x':
#             reverse = True
#         elif subj.startswith('y') and obj.startswith('y') and subj > obj:       # y1 v.s. y0
#             reverse = True
#         if reverse:
#             subj, obj = obj, subj
#             pred = fb_helper.inverse_predicate(pred)
#         edge_dict[subj] = (pred, obj)
#
#     # Step 2: Search for start entity, and traverse the schema.
#     for start_subj in edge_dict:
#         if is_intermediate_node(start_subj):
#             continue
#         # now traverse the tree
#         path = [start_subj]             # store the [e, p1, p2, ...]
#         cur_subj = start_subj
#         while cur_subj != 'x':
#             pred, obj = edge_dict[cur_subj]
#             path.append(pred)
#             cur_subj = obj  # go to next node in the path
#         path_str = '-->'.join(path)
#         path_tup_list.append((path, path_str))
#
#     path_tup_list.sort(key=lambda x: x[1])            # sort by alphabet order of the path name
#     path_list = [tup[0] for tup in path_tup_list]
#     path_list_str = '\t'.join([tup[1] for tup in path_tup_list])
#
#     sc = Schema(path_list=path_list)
#     return sc, path_list_str
