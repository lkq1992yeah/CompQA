"""
Simulation for modeling contest 2017 problem E
Implemented by Xusheng Luo
"""

from floyd import Floyd

import copy

class PathPlanner(object):

    def __init__(self):
        self.mapA = Floyd(fp='./distance.txt', car_type='A')
        self.mapB = Floyd(fp='./distance.txt', car_type='B')
        self.mapC = Floyd(fp='./distance.txt', car_type='C')
        car_name = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
        self.car_name_idx_map = dict()
        self.car_idx_name_map = dict()
        for idx, name in enumerate(car_name):
            self.car_idx_name_map[idx] = name
            self.car_name_idx_map[name] = idx

        self.phase_1_ret = dict()
        self.phase_2_ret = dict()
        self.phase_3_ret = dict()

        self.visited_fs = set()
        self.c_set = list()
        self.phase_1_end_time = 0

    def run_phase_1(self):

        # ret, each car's destination & detailed path in three phases
        # {car_idx: [st_idx, des_idx, [[path_node_idx, [arrival_time, leave_time]]]}

        # print("\n[log] ============ Phase 1 starts ============ \n")

        D1FA_cands = self.mapA.get_nearest_f('D1', 12)
        D1FB_cands = self.mapB.get_nearest_f('D1', 12)
        D1FC_cands = self.mapC.get_nearest_f('D1', 12)
        D2FA_cands = self.mapA.get_nearest_f('D2', 12)
        D2FB_cands = self.mapB.get_nearest_f('D2', 12)
        D2FC_cands = self.mapC.get_nearest_f('D2', 12)

        # ------------------------------ Search for 12 F points ------------------------------ #

        # --- D1 --- #
        # obviously the nearest 6 F should belong to car type C
        for idx, node_pair in enumerate(D1FC_cands[:6]):
            car_idx = idx + 12
            node_idx = node_pair[0]
            node_time = node_pair[1]
            path = self.mapC.get_path_and_time(self.mapC.idx_name_map[node_idx], 'D1', 0,
                                               self.car_idx_name_map[car_idx])
            assert node_time == path[-1][1][1]
            self.phase_1_ret[car_idx] = [node_idx, 0, path]
            self.visited_fs.add(node_idx)

        # search for B, non-self.visited_fs 3 F
        i = 0
        car_idx = 6
        while i < 12 and car_idx < 9:
            node_idx, node_time = D1FB_cands[i]
            if node_idx in self.visited_fs:
                i += 1
                continue
            path = self.mapB.get_path_and_time(self.mapB.idx_name_map[node_idx], 'D1', 0,
                                               self.car_idx_name_map[car_idx])
            assert node_time == path[-1][1][1]
            self.phase_1_ret[car_idx] = [node_idx, 0, path]
            self.visited_fs.add(node_idx)
            i += 1
            car_idx += 1

        # search for A, non-self.visited_fs 3 F
        i = 0
        car_idx = 0
        while i < 12 and car_idx < 3:
            node_idx, node_time = D1FA_cands[i]
            if node_idx in self.visited_fs:
                i += 1
                continue
            path = self.mapA.get_path_and_time(self.mapA.idx_name_map[node_idx], 'D1', 0,
                                               self.car_idx_name_map[car_idx])
            assert node_time == path[-1][1][1]
            self.phase_1_ret[car_idx] = [node_idx, 0, path]
            self.visited_fs.add(node_idx)
            i += 1
            car_idx += 1

        # --- D2 --- #
        # obviously the nearest 6 F should belong to car type C
        for idx, node_pair in enumerate(D2FC_cands[:6]):
            car_idx = idx + 18
            node_idx = node_pair[0]
            node_time = node_pair[1]
            path = self.mapC.get_path_and_time(self.mapC.idx_name_map[node_idx], 'D2', 0,
                                               self.car_idx_name_map[car_idx])
            assert node_time == path[-1][1][1]
            self.phase_1_ret[car_idx] = [node_idx, 1, path]
            self.visited_fs.add(node_idx)

        # search for B, non-self.visited_fs 3 F
        i = 0
        car_idx = 9
        while i < 12 and car_idx < 12:
            node_idx, node_time = D2FB_cands[i]
            if node_idx in self.visited_fs:
                i += 1
                continue
            path = self.mapB.get_path_and_time(self.mapB.idx_name_map[node_idx], 'D2', 0,
                                               self.car_idx_name_map[car_idx])
            assert node_time == path[-1][1][1]
            self.phase_1_ret[car_idx] = [node_idx, 1, path]
            self.visited_fs.add(node_idx)
            i += 1
            car_idx += 1

        # search for A, non-self.visited_fs 3 F
        i = 0
        car_idx = 3
        while i < 12 and car_idx < 6:
            node_idx, node_time = D2FA_cands[i]
            if node_idx in self.visited_fs:
                i += 1
                continue
            path = self.mapA.get_path_and_time(self.mapA.idx_name_map[node_idx], 'D2', 0,
                                               self.car_idx_name_map[car_idx])
            assert node_time == path[-1][1][1]
            self.phase_1_ret[car_idx] = [node_idx, 1, path]
            self.visited_fs.add(node_idx)
            i += 1
            car_idx += 1

        # print("Phase 1 ret: %s" % self.phase_1_ret)

        # ------------------------------ check conflicts ------------------------------ #

        # print("\n[log] Checking conflicts in Phase 1...\n")
        conflicts = self.check_conflict(self.phase_1_ret)
        while len(conflicts) != 0:
            self.phase_1_ret = self.solve_conflict(conflicts, self.phase_1_ret)
            conflicts = self.check_conflict(self.phase_1_ret)

        # ------------------------------ find max time ------------------------------ #

        max_time = 0
        for car_idx, [_, _, path] in self.phase_1_ret.items():
            if path[-1][1][0] > max_time:
                max_time = path[-1][1][0]

        self.phase_1_end_time = max_time
        # print("[phase1] Max time: %.4f." % max_time)

        # ------------------------------ reverse ------------------------------ #

        for car_idx in self.phase_1_ret.keys():
            [st, ed, path] = self.phase_1_ret[car_idx]
            self.phase_1_ret[car_idx][0] = ed
            self.phase_1_ret[car_idx][1] = st
            for i in range(len(path)):
                [_, [arr, lea]] = path[i]
                path[i][1][0] = max_time - lea
                path[i][1][1] = max_time - arr
            path.reverse()

    def run_phase_2(self, c_set, sudden_change):

        # ret, each car's destination & detailed path in three phases
        # {car_idx: (start_idx, destination_idx, [(path_node_idx, arrival time stamp)])}

        # print("\n[log] ============ Phase 2 starts ============ \n")

        self.c_set = c_set
        self.phase_2_ret = dict()

        f_starts = {x: y[1] for x, y in self.phase_1_ret.items()}

        for car_idx, f_idx in f_starts.items():
            if car_idx in range(0, 6):
                cands = self.mapA.get_nearest_z(self.mapA.idx_name_map[f_idx])
                z_idx = cands[0][0]
                path = self.mapA.get_path_and_time(self.mapA.idx_name_map[f_idx],
                                                   self.mapA.idx_name_map[z_idx],
                                                   self.phase_1_end_time,
                                                   self.car_idx_name_map[car_idx])
                self.phase_2_ret[car_idx] = [f_idx, z_idx, path]
            elif car_idx in range(6, 12):
                cands = self.mapB.get_nearest_z(self.mapB.idx_name_map[f_idx])
                z_idx = cands[0][0]
                path = self.mapB.get_path_and_time(self.mapB.idx_name_map[f_idx],
                                                   self.mapB.idx_name_map[z_idx],
                                                   self.phase_1_end_time,
                                                   self.car_idx_name_map[car_idx])
                self.phase_2_ret[car_idx] = [f_idx, z_idx, path]
            elif car_idx not in self.c_set:
                cands = self.mapC.get_nearest_z(self.mapC.idx_name_map[f_idx])
                z_idx = cands[0][0]
                path = self.mapC.get_path_and_time(self.mapC.idx_name_map[f_idx],
                                                   self.mapC.idx_name_map[z_idx],
                                                   self.phase_1_end_time,
                                                   self.car_idx_name_map[car_idx])
                self.phase_2_ret[car_idx] = [f_idx, z_idx, path]

            # ret = ["%s(%d, %.2f min)" % (self.mapA.idx_name_map[f[0]], f[0], f[1]) for f in cands]
            # print("[%s] starts from %s, nearest-Z: %s" % (self.car_idx_name_map[car_idx],
            #                                               self.mapA.idx_name_map[f_idx],
            #                                               '|'.join(ret)))

        # ------------------------------ check conflicts ------------------------------ #

        # print("\n[log] Checking conflicts in Phase 2...\n")
        conflicts = self.check_conflict(self.phase_2_ret)
        while len(conflicts) != 0:
            self.phase_2_ret = self.solve_conflict(conflicts, self.phase_2_ret)
            conflicts = self.check_conflict(self.phase_2_ret)

        [j1, j2, j3] = sudden_change
        self.phase_2_ret[self.c_set[0]] = [j1, j1, [[j1, [self.phase_1_end_time, self.phase_1_end_time]]]]
        self.phase_2_ret[self.c_set[1]] = [j2, j2, [[j2, [self.phase_1_end_time, self.phase_1_end_time]]]]
        self.phase_2_ret[self.c_set[2]] = [j3, j3, [[j3, [self.phase_1_end_time, self.phase_1_end_time]]]]

    def run_phase_3(self):

        # print("\n[log] ============ Phase 3 starts ============\n")

        self.phase_3_ret = dict()
        self.visited_fs_reuse = copy.deepcopy(self.visited_fs)

        # ------------------------------ manage queue in Z ------------------------------ #

        queue_z = dict()
        for car_idx, [_, des_idx, path] in self.phase_2_ret.items():
            if car_idx in self.c_set:
                continue
            if des_idx not in queue_z:
                queue_z[des_idx] = list()
            queue_z[des_idx].append((car_idx, path[-1][1][0]))

        # print("[phase3] Queue at Zs: %s\n" % queue_z)

        phase_3_st_time = dict()
        phase_3_st_time[self.c_set[0]] = (self.phase_2_ret[self.c_set[0]][0], self.phase_1_end_time)
        phase_3_st_time[self.c_set[1]] = (self.phase_2_ret[self.c_set[1]][0], self.phase_1_end_time)
        phase_3_st_time[self.c_set[2]] = (self.phase_2_ret[self.c_set[2]][0], self.phase_1_end_time)
        for z_idx, car_idx_time_pair in queue_z.items():
            car_idx_time_pair.sort(lambda a, b: cmp(a[1], b[1]))
            pre_time = -1
            for i, (car_idx, time) in enumerate(car_idx_time_pair):
                if i == 0:
                    phase_3_st_time[car_idx] = (z_idx, time + 10)
                    pre_time = time + 10
                else:
                    if time - pre_time >= 0:
                        phase_3_st_time[car_idx] = (z_idx, time + 10)
                        pre_time = time + 10
                    else:
                        phase_3_st_time[car_idx] = (z_idx, pre_time + 10)
                        pre_time = pre_time + 10

        # print("[phase3] Start time from Zs: %s\n" % phase_3_st_time)

        # ------------------------------ search from Z to F again ------------------------------ #

        cands_map = dict()
        priority_list = list()
        for car_idx, (z_idx, st_time) in phase_3_st_time.items():
            if car_idx in range(0, 6):
                cands_map[car_idx] = self.mapA.get_nearest_f(self.mapA.idx_name_map[z_idx], 60)
                priority_list.append((car_idx, st_time / 9))
            elif car_idx in range(6, 12):
                cands_map[car_idx] = self.mapB.get_nearest_f(self.mapB.idx_name_map[z_idx], 60)
                priority_list.append((car_idx, st_time / 7))
            else:
                cands_map[car_idx] = self.mapC.get_nearest_f(self.mapC.idx_name_map[z_idx], 60)
                priority_list.append((car_idx, st_time / 6))

        priority_list.sort(lambda a, b: -cmp(a[1], b[1]))
        priority_show = [self.car_idx_name_map[x[0]] for x in priority_list]
        # print("[phase3] Priority list of choosing Fs: %s\n" % priority_show)

        for (car_idx, _) in priority_list:
            cands = cands_map[car_idx]
            for (f_idx, time) in cands:
                if f_idx not in self.visited_fs_reuse:
                    (z_idx, st_time) = phase_3_st_time[car_idx]
                    if car_idx in range(0, 6):
                        path = self.mapA.get_path_and_time(self.mapA.idx_name_map[z_idx],
                                                           self.mapA.idx_name_map[f_idx],
                                                           st_time,
                                                           self.car_idx_name_map[car_idx])
                    elif car_idx in range(6, 12):
                        path = self.mapB.get_path_and_time(self.mapB.idx_name_map[z_idx],
                                                           self.mapB.idx_name_map[f_idx],
                                                           st_time,
                                                           self.car_idx_name_map[car_idx])
                    else:
                        path = self.mapC.get_path_and_time(self.mapC.idx_name_map[z_idx],
                                                           self.mapC.idx_name_map[f_idx],
                                                           st_time,
                                                           self.car_idx_name_map[car_idx])
                    self.phase_3_ret[car_idx] = [z_idx, f_idx, path]
                    self.visited_fs_reuse.add(f_idx)
                    break

        # ------------------------------ check conflicts ------------------------------ #

        # print("\n[log] Checking conflicts in Phase 3...\n")
        conflicts = self.check_conflict(self.phase_3_ret)
        while len(conflicts) != 0:
            self.phase_3_ret = self.solve_conflict(conflicts, self.phase_3_ret)
            conflicts = self.check_conflict(self.phase_3_ret)

        # ------------------------------ find max time ------------------------------ #

        max_time = 0
        for car_idx, [_, _, path] in self.phase_3_ret.items():
            if path[-1][1][0] > max_time:
                max_time = path[-1][1][0]
        # print("[phase3] Max time: %.4f." % max_time)

    def check_conflict(self, ret_dict):
        # print("\n[log] Checking conflicts...\n")
        road_detail = dict()
        conflicts = list()
        for car_idx, [st_idx, des_idx, path] in ret_dict.items():
            for i in range(len(path) - 1):
                node_interval = (path[i][0], path[i + 1][0])
                time_interval = (path[i][1][1], path[i + 1][1][0])
                if node_interval not in road_detail:
                    road_detail[node_interval] = list()
                road_detail[node_interval].append((car_idx, time_interval))
        # print("[log] Road details loaded.")

        for node_interval, time_intervals in road_detail.items():
            # forward conflict
            time_intervals.sort(lambda a, b: cmp(a[1][0], b[1][0]))
            flag = True
            for i in range(0, len(time_intervals)-1):
                if not flag:
                    break
                car_idx, (st, ed) = time_intervals[i]
                for j in range(i+1, len(time_intervals)):
                    next_car_idx, (next_st, next_ed) = time_intervals[j]
                    if ed - next_ed > 0.01:
                        node_interval_show = "<%s, %s>" % (self.mapA.idx_name_map[node_interval[0]],
                                                           self.mapA.idx_name_map[node_interval[1]])
                        time_intervals_show = ["%s(%.2f->%.2f)" % (self.car_idx_name_map[x[0]],
                                                                   x[1][0], x[1][1]) for x in time_intervals]
                        # print("[log] forward conflict %s: %s" % (node_interval_show,
                        #                                          '|'.join(time_intervals_show)))
                        conflicts.append((node_interval, time_intervals))
                        flag = False
                        break

            # backward conflict
            if self.is_main_road(node_interval[0], node_interval[1]):
                continue
        return conflicts

    def solve_conflict(self, conflicts, ret_dict):
        """
        :param conflicts: [(node_interval, [(car_idx, time_interval)])]
        :param ret_dict: {car_idx: (st_idx, des_idx, [(node_idx, time_stamp)])} 
        :return: 
        """
        # print("\n[log] Solving conflict...\n")
        for (node_interval, time_intervals) in conflicts:
            (st_node, ed_node) = node_interval
            flag = True
            for i in range(0, len(time_intervals)-1):
                if not flag:
                    break
                car_idx, (st, ed) = time_intervals[i]
                for j in range(i+1, len(time_intervals)):
                    next_car_idx, (next_st, next_ed) = time_intervals[j]
                    if next_ed < ed:
                        # print("[conflict] Car %s(%.2f, %.2f) conflicts with car %s(%.2f, %.2f)"
                        #       " at <%s, %s>" %
                        #       (self.car_idx_name_map[car_idx], st, ed,
                        #        self.car_idx_name_map[next_car_idx], next_st, next_ed,
                        #        self.mapA.idx_name_map[st_node],
                        #        self.mapA.idx_name_map[ed_node]))
                        # solve
                        st_margin = next_st - st
                        ed_margin = ed - next_ed
                        if st_margin > ed_margin:
                            # next car wait
                            path = ret_dict[next_car_idx][2]
                            pos = 0
                            while path[pos][0] != st_node:
                                pos += 1
                            path[pos][1][1] += ed_margin
                            path[pos+1][1][0] += ed_margin
                            for k in range(pos+1, len(path)):
                                if path[k][1][0] - path[k][1][1] > 0.01:
                                    path[k][1][1] = path[k][1][0]
                                    if k != len(path)-1:
                                        path[k+1][0] += (path[k][1][0]-path[k][1][0])
                            # print("[update] Car %s: %s" % (self.car_idx_name_map[next_car_idx],
                            #                                ret_dict[next_car_idx]))
                        else:
                            # pre car wait
                            path = ret_dict[car_idx][2]
                            pos = 0
                            while path[pos][0] != st_node:
                                pos += 1
                            path[pos][1][1] += st_margin
                            path[pos+1][1][0] += st_margin
                            for k in range(pos+1, len(path)):
                                if path[k][1][0] - path[k][1][1] > 0.01:
                                    path[k][1][1] = path[k][1][0]
                                    if k != len(path)-1:
                                        path[k+1][0] += (path[k][1][0]-path[k][1][0])
                            # print("[update] car %s: %s" % (self.car_idx_name_map[car_idx],
                            #                                ret_dict[car_idx]))
                        flag = False
                        break
        return ret_dict

    @staticmethod
    def is_main_road(i, j):
        if i - j == 1 and (i in range(69, 79) or i in range(80, 88)) \
                or i - j == -1 and (i in range(68, 78) or i in range(79, 87)):
            return True
        else:
            return False

    def summarize(self):
        # print("\n[summary] ========================== Final Result ========================\n")
        for car_idx, [st, ed, path] in self.phase_1_ret.items():
            show = ["%s(%d, [%.2f, %.2f]~min)" % (self.mapA.idx_name_map[x[0]], x[0], x[1][0], x[1][1]) for x in path]
            print("[%s] Path from %s(%d) to %s(%d) : [ %s ]." %
                  (self.car_idx_name_map[car_idx],
                   self.mapA.idx_name_map[st], st,
                   self.mapA.idx_name_map[ed], ed,
                   " -> ".join(show)))

        print("\n")
        for car_idx, [st, ed, path] in self.phase_2_ret.items():
            if car_idx in self.c_set:
                continue
            show = ["%s(%d, [%.2f, %.2f]~min)" % (self.mapA.idx_name_map[x[0]], x[0], x[1][0], x[1][1]) for x in path]
            print("[%s] Path from %s(%d) to %s(%d) : [ %s ]." %
                  (self.car_idx_name_map[car_idx],
                   self.mapA.idx_name_map[st], st,
                   self.mapA.idx_name_map[ed], ed,
                   " -> ".join(show)))
        print("\n")
        for car_idx, [st, ed, path] in self.phase_3_ret.items():
            show = ["%s(%d, [%.2f, %.2f]~min)" % (self.mapA.idx_name_map[x[0]], x[0], x[1][0], x[1][1]) for x in path]
            print("[%s] Path from %s(%d) to %s(%d) : [ %s ]." %
                  (self.car_idx_name_map[car_idx],
                   self.mapA.idx_name_map[st], st,
                   self.mapA.idx_name_map[ed], ed,
                   " -> ".join(show)))
        print("\n")

        total = 0
        start = 0
        wait = 0
        avg_max = 0
        avg_min = 99999999
        car_avg = list()
        for car_idx in range(24):
            path_1 = self.phase_1_ret[car_idx][2]
            start += path_1[0][1][1]
            path_2 = self.phase_2_ret[car_idx][2]
            path_3 = self.phase_3_ret[car_idx][2]
            wait += path_3[0][1][1] - path_2[-1][1][0]
            total += path_3[-1][1][0]
            avg = path_3[-1][1][0] - path_1[0][1][1] - (path_3[0][1][1] - path_2[-1][1][0])
            avg_max = max(avg_max, avg)
            avg_min = min(avg_min, avg)
            car_avg.append((self.car_idx_name_map[car_idx], avg))
        car_avg.sort(lambda a, b: -cmp(a[1], b[1]))
        # print("car_avg: %s" % car_avg)
        # print("%.4f(%.4f, %.4f, %.4f)" % (total-start-wait, total, start, wait))
        # print("max_avg: %.4f, min_avg: %.4f" % (avg_max, avg_min))
        return total-start-wait

if __name__ == '__main__':
    planner = PathPlanner()
    planner.run_phase_1()
    min_ret = 999999
    min_way = list()
    for i in range(12, 24):
        t = range(12, 24)
        t.remove(i)
        for j in t:
            t2 = copy.deepcopy(t)
            t2.remove(j)
            for k in t2:
                init = [i, j, k]
                cands = ['J04', 'J06', 'J08', 'J13', 'J14', 'J15']
                for x1 in cands:
                    tmp = copy.deepcopy(cands)
                    tmp.remove(x1)
                    for x2 in tmp:
                        tmp2 = copy.deepcopy(tmp)
                        tmp2.remove(x2)
                        for x3 in tmp2:
                            planner.run_phase_2(init,
                                                [planner.mapA.name_idx_map[x0] for x0 in [x1, x2, x3]])
                            planner.run_phase_3()
                            ret = planner.summarize()
                            # print("[log] choice %s : %.4f" % ([x1, x2, x3], ret))
                            if ret < min_ret:
                                min_ret = ret
                                min_way = [i, j, k, x1, x2, x3]
                                print("[log] Better choice: %s(%.4f)" % (min_way, min_ret))
    print("[log] Best choice: %s(%.4f)" % (min_way, min_ret))
