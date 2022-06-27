from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    name = 'VID'
    if name == 'GOT-10k':
        root_dir = os.path.expanduser('~/data/GOT-10k')
        seqs = GOT10k(root_dir, subset='train', return_meta=True)
    elif name == 'VID':
        root_dir = 'E:\\Reseach\\Dataset\\Tracking Dataset\\ILSVRC\\ILSVRC2015_VID\\imagenet2015\\imagenet2015\\ILSVRC\\'
        seqs = ImageNetVID(root_dir, subset=('train', 'val'))
    elif name == 'VOT':
        root_dir = 'E:\\Reseach\\Dataset\\Tracking Dataset\\VOT2017\\'
        seqs = VOT(root_dir, download=False)


    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
