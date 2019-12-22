from multiprocessing import Lock, Process, Pool, Manager

from kangqi.task.compQA.candgen_acl18.query_service import QueryService

from kangqi.util.LogUtil import LogInfo
import time
import random


class TestObj:

    def test_unpack(self, args):
        return self.test(*args)

    def test(self, a, b, c, lock):
        lock.acquire()
        for _ in range(c):
            LogInfo.logs('a = %d, b = %d', a, b)
            time.sleep(random.random())
        lock.release()


def real_test_unpack(args):
    return real_test(*args)


def real_test(obj, a, b, c, lock):
    obj.test(a, b, c, lock)


class AnoterTestObj:

    def __init__(self, lock):
        self.lock = lock

    def test(self, query_srv, a, b, c):
        self.lock.acquire()
        for _ in range(c):
            # if query_srv is not None:
            #     LogInfo.logs(query_srv.query_prefix)
            LogInfo.logs('a = %d, b = %d', a, b)
            time.sleep(random.random())
        self.lock.release()


def another_real_test_unpack(args):
    return another_real_test(*args)


def another_real_test(ano_obj, query_srv, a, b, c):
    ano_obj.test(query_srv, a, b, c)




def main():
    m = Manager()
    lock = m.Lock()
    p = Pool(processes=5)
    # iterable = [1, 2, 3, 4, 5]
    # my_func = partial(target, lock)
    # p.map(func=my_func, iterable=iterable)

    obj = TestObj()
    # ano_obj = AnoterTestObj(lock=lock)

    query_srv = QueryService(vb=1)  # QueryService with multi-thread control
    # query_srv = None

    args = []
    for task in range(3):
        args.append([obj, task, task*2, 5, lock])
    LogInfo.logs(args)
    p.map(func=real_test_unpack, iterable=args)

    p.close()
    p.join()

    # for idx in range(10):
    #     Process(target=test, args=(idx, idx * 2, 5, lock)).start()


if __name__ == '__main__':
    main()
