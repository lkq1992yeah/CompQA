"""
Floyd algorithm for modeling contest 2017 problem E
Load from file in a format of adjacent matrix
"""

import copy

class Floyd(object):

    def __init__(self, fp, car_type):
        self.car_type = car_type
        self.name_idx_map = dict()
        self.idx_name_map = dict()
        self.car_v1_map = {'A': 70, 'B': 60, 'C': 50}
        self.car_v2_map = {'A': 45, 'B': 35, 'C': 30}
        self.mat = list()
        self.mid = list()
        self.path = list()
        print("[%s] Loading adjacent matrix from %s..." % (self.car_type, fp))
        with open(fp, 'r') as fin:
            cnt = 0
            for line in fin:
                spt = line.strip().split()
                assert len(spt) == 131
                name = spt[0]
                self.name_idx_map[name] = cnt
                self.idx_name_map[cnt] = name
                row = list()
                for i, d in enumerate(spt[1:]):
                    if i == cnt:
                        row.append(0)
                    elif d == '0':
                        row.append(-1)
                    else:
                        row.append(float(d))
                self.mat.append(row)
                self.mid.append([-1 for _ in range(130)])
                cnt += 1
        self.n = len(self.mat)
        self.dist_mat = copy.deepcopy(self.mat)

        fout = open('edges.txt', 'w')
        for i in range(self.n):
            for j in range(self.n):
                if self.mat[i][j] > 0:
                    fout.write("%s => %s\n" % (self.idx_name_map[i],
                                             self.idx_name_map[j]))
                    fout.write("%s => %s\n" % (self.idx_name_map[j],
                                             self.idx_name_map[i]))
        fout.close()

        for i in range(self.n):
            for j in range(self.n):
                if self.mat[i][j] > 0:
                    self.dist_mat[j][i] = self.dist_mat[i][j]
                    self.mat[j][i] = self.mat[i][j]

        for i in range(self.n):
            for j in range(self.n):
                if self.mat[i][j] > 0:
                    if i-j==1 and (i in range(69, 79) or i in range(80, 88)) \
                            or i-j==-1 and (i in range(68, 78) or i in range(79, 87)):
                        self.mat[i][j] /= self.car_v1_map[car_type]
                    else:
                        self.mat[i][j] /= self.car_v2_map[car_type]
                    self.mat[i][j] *= 60

        print("[%s] Adjacent matrix loaded. (%d, %d)" % (self.car_type, self.n, self.n))
        self.original_time_mat = copy.deepcopy(self.mat)
        self.original_dist_mat = copy.deepcopy(self.dist_mat)
        self.run()

    def run(self):
        print("[%s] Running floyd algorithm..." % self.car_type)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.mat[i][k] == -1 or self.mat[k][j] == -1:
                        continue
                    if self.mat[i][j] == -1 or \
                            self.mat[i][j] > self.mat[i][k] + self.mat[k][j]:
                        self.mat[i][j] = self.mat[i][k] + self.mat[k][j]
                        self.dist_mat[i][j] = self.dist_mat[i][k] + self.dist_mat[k][j]
                        self.mid[i][j] = k
        print("[%s] Floyd algorithm done." % self.car_type)

    def get_time_neighbours(self, name):
        idx = self.name_idx_map[name]
        ret = list()
        for i in range(self.n):
            if self.original_time_mat[idx][i] > 0:
                ret.append((self.idx_name_map[i], self.original_time_mat[idx][i]))
        print("[ret] Time neighbours of %s are %s." % (name, ret))

    def get_dist_neighbours(self, name):
        idx = self.name_idx_map[name]
        ret = list()
        for i in range(self.n):
            if self.original_dist_mat[idx][i] > 0:
                ret.append((self.idx_name_map[i], self.original_dist_mat[idx][i]))
        print("[ret] Dist neighbours of %s are %s." % (name, ret))

    def get_time(self, name_i, name_j):
        i = self.name_idx_map[name_i]
        j = self.name_idx_map[name_j]
        assert i < self.n and j < self.n
        print("[%s] Shortest time between %s(%d) and %s(%d) is %.2f min." %
              (self.car_type, name_i, i, name_j, j, self.mat[i][j]))
        return self.mat[i][j]

    def get_distance(self, name_i, name_j):
        i = self.name_idx_map[name_i]
        j = self.name_idx_map[name_j]
        assert i < self.n and j < self.n
        print("[%s] Shortest distance between %s(%d) and %s(%d) is %.2f km." %
              (self.car_type, name_i, i, name_j, j, self.dist_mat[i][j]))
        return self.dist_mat[i][j]

    def get_path(self, name_i, name_j):
        i = self.name_idx_map[name_i]
        j = self.name_idx_map[name_j]
        assert i < self.n and j < self.n
        self.path = [i]
        self.get_path_by_idx(i, j)
        # ret = [self.idx_name_map[node] for node in self.path]
        # print("[%s] Path from %s(%d) to %s(%d) : [%s]." %
        #       (self.car_type, name_i, i, name_j, j, " -> ".join(ret)))
        return self.path

    def get_path_and_time(self, name_i, name_j, start_time, car_name):
        path = self.get_path(name_i, name_j)
        i = self.name_idx_map[name_i]
        j = self.name_idx_map[name_j]
        time = [start_time + self.get_distance_by_idx(i, node) for node in path]
        ret = [[x, [y, y]] for x, y in zip(path, time)]
        show = ["%s(%d, [%.2f, %.2f]~min)" % (self.idx_name_map[x[0]], x[0], x[1][0], x[1][1]) for x in ret]
        # print("[%s] Path from %s(%d) to %s(%d) : [ %s ]." %
        #       (car_name, name_i, i, name_j, j, " -> ".join(show)))
        return ret

    def get_distance_by_idx(self, i, j):
        assert i < self.n and j < self.n
        return self.mat[i][j]

    def get_path_by_idx(self, i, j):
        assert i < self.n and j < self.n
        if i == j:
            return
        elif self.mid[i][j] == -1:
            self.path.append(j)
        else:
            self.get_path_by_idx(i, self.mid[i][j])
            self.get_path_by_idx(self.mid[i][j], j)

    def get_nearest_f(self, name, num):
        idx = self.name_idx_map[name]
        row = self.mat[idx]
        f_dist = [(i, row[i]) for i in range(8, 68)]
        f_dist.sort(lambda a, b: cmp(a[1], b[1]))
        ret = ["%s(%d, %.2f min)" % (self.idx_name_map[f[0]], f[0], f[1]) for f in f_dist[:num]]
        # print("[%s] nearest F-distance from %s: %s." % (self.car_type, name, ret))
        return f_dist[:num]

    def get_nearest_z(self, name, extra_1=None, extra_2=None):
        idx = self.name_idx_map[name]
        row = self.mat[idx]
        z_range = range(2, 8)
        if extra_1 is not None:
            z_range.append(self.name_idx_map[extra_1])
        if extra_2 is not None:
            z_range.append(self.name_idx_map[extra_2])
        z_dist = [(i, row[i]) for i in z_range]
        z_dist.sort(lambda a, b: cmp(a[1], b[1]))
        ret = ["%s(%d, %.2f min)" % (self.idx_name_map[f[0]], f[0], f[1]) for f in z_dist]
        # print("[%s] nearest F-distance from %s: %s." % (self.car_type, name, ret))
        return z_dist

if __name__ == '__main__':
    testA = Floyd('./distance.txt', 'A')
    testB = Floyd('./distance.txt', 'B')
    testC = Floyd('./distance.txt', 'C')
    testA.get_dist_neighbours('D1')
    testB.get_dist_neighbours('D1')
    testC.get_dist_neighbours('D1')
    testA.get_time_neighbours('D1')
    testB.get_time_neighbours('D1')
    testC.get_time_neighbours('D1')
    testA.get_distance('J09', 'J08')
    testB.get_distance('D1', 'J07')
    testC.get_distance('D1', 'J07')
    testA.get_time('D1', 'J07')
    testB.get_time('D1', 'J07')
    testC.get_time('D1', 'J07')

    # testA.get_nearest_f('D1', 6)
    # testB.get_nearest_f('D1', 6)
    # testC.get_nearest_f('D1', 6)
    # testA.get_nearest_f('D2', 6)
    # testB.get_nearest_f('D2', 6)
    # testC.get_nearest_f('D2', 6)
    #
    # testA.get_nearest_z('F42', 6)
    # testB.get_nearest_z('F29', 6)
    # testC.get_nearest_z('F43', 6)
    # testA.get_nearest_z('D1', 6)
    # testB.get_nearest_z('D2', 6)
    # testC.get_nearest_z('J01', 6)

