dphmm
=====
do make to compile.

A typical command should look like this:
/usr/users/chiaying/phone_learning/decode/decode_to_pg -i /data/scratch/jlee/posteriors/possible3696_bound_annealing/snapshot_100 -d /usr/users/chiaying/TIMIT/SUMMIT/25ms/raw_mfcc_pca_whitened/si806-b-mwdk0.raw -o /usr/users/chiaying/keyword/pgs/hsmm100/train/si806-b-mwdk0.mc -d 39 -m 2 -t 0 -a 0 -b 0

The model file can be found in model/snapshot.
