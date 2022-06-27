from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/backup/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.test_mode = 1

    #root_dir = os.path.expanduser('E:/Reseach/Dataset/Tracking Dataset/OTB100/')
    #e = ExperimentOTB(root_dir, version='tb100')

    root_dir = os.path.expanduser('E:/Reseach/Dataset/Tracking Dataset/LaSOT/LaSOTBenchmark')
    e = ExperimentLaSOT(root_dir,subset='test')

    #root_dir = os.path.expanduser('E:/Reseach/Dataset/Tracking Dataset/GOT-10k/')
    #e = ExperimentGOT10k(root_dir, subset='val')

    e.run(tracker)
    e.report([tracker.name])
