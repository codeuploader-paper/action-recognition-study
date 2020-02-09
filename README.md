# Action Recognition Study


## training script

Training an I3D on the Kine
```
python3 train.py --datadir /path/to/folder --threed_data \
--dataset kinetics400 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d -b 64 -j 36
```

## training script, distributed version on a single node

Not tested for multiple nodes yet.
The distributed version supports synchorize the running mean and var in the batchnorm layer

```
python3 train_dist.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36 --sync-bn --multiprocessing-distributed
```

## training script, distributed version on multi-node

It works but not validate the accuracy yet.
The distributed version supports synchorize the running mean and var in the batchnorm layer

Launch the following command on each machine individually, E.g. two nodes:
The first node:
```
python3 train_dist.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36 --sync-bn --multiprocessing-distributed \
--world-size 2 --rank 0 --dist-url tcp://URL:PORT
```

The second node (put the same `tcp://URL:PORT` to the first node, since the first node will be used as parameter server):
```
python3 train_dist.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36 --sync-bn --multiprocessing-distributed \
--world-size 2 --rank 1 --dist-url tcp://URL:PORT
```

