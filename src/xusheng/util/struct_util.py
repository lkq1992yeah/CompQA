"""
Top k ranked list
Fixed-size, pair-tuple-element, priority queue
"""


class TopKRankedList(object):

    def __init__(self, k):
        self.max_size = k
        self.data = list()

    def push(self, obj):
        if self._get_size() == self.max_size and self._get_last()[1] >= obj[1]:
            return
        if self._get_size() < self.max_size:
            self.data.append(obj)
        else:
            self.data[-1] = obj
        i = self._get_size()-1
        while i > 0 and self.data[i][1] > self.data[i-1][1]:
            tmp = self.data[i]
            self.data[i] = self.data[i-1]
            self.data[i-1] = tmp
            i -= 1

    def top(self):
        if self._get_size() != 0:
            return self.data[0]
        else:
            return None

    def top_names(self):
        ret = list()
        for tup in self.data:
            ret.append(tup[0])
        return ret

    def clear(self):
        self.data.clear()

    def set_size(self, k):
        self.max_size = k

    def _get_size(self):
        return len(self.data)

    def _get_last(self):
        return self.data[-1]

if __name__ == "__main__":
    from util.log_util import LogInfo
    rank_list = TopKRankedList(5)
    rank_list.push(("a", 3))
    LogInfo.logs(rank_list.data)
    rank_list.push(("b", 4))
    LogInfo.logs(rank_list.data)
    rank_list.push(("c", 1))
    LogInfo.logs(rank_list.data)
    rank_list.push(("d", 2))
    LogInfo.logs(rank_list.data)
    rank_list.push(("e", 3))
    LogInfo.logs(rank_list.data)
    rank_list.push(("f", 0))
    LogInfo.logs(rank_list.data)
    rank_list.push(("g", 1.2))
    LogInfo.logs(rank_list.data)
    rank_list.push(("h", 12))
    LogInfo.logs(rank_list.data)
