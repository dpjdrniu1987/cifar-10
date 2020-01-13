# -*- coding: utf-8 -*-
import os
import tarfile

if not os.path.exists("data/cifar-10-batches-py"):
    tfile=tarfile.open("data/cifar-10-python.tar.gz",'r:gz')
    result=tfile.extractall('data/')
    print("Extracted to ./data/cifar-10-batches-py")
else:
    print('Directory already exists.')

