import pyltr
from kangqi.util.LogUtil import LogInfo

data_dir = 'data/pyltr/MQ2007/Fold1'

with open(data_dir + '/train.txt') as trainfile, \
        open(data_dir + '/vali.txt') as valifile, \
        open(data_dir + '/test.txt') as evalfile:
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

metric = pyltr.metrics.NDCG(k=10)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=20)        # 250

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=10,    # 1000
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)

Epred = model.predict(EX)
LogInfo.logs('Epred: %s', len(Epred))
LogInfo.logs(Epred)

print 'Random ranking:', metric.calc_mean_random(Eqids, Ey)
print 'Our model:', metric.calc_mean(Eqids, Ey, Epred)
