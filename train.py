from __future__ import absolute_import, division, print_function
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
from trainer import Trainer
from options import MonodepthOptions
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def printArgs(args):
    argsDict = args.__dict__
    print("Args:")
    print('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
            print(eachArg + ' : ' + str(value) + '\n')
    print('------------------- end -------------------')

options = MonodepthOptions()
opts = options.parse()

if __name__ == "__main__":
    printArgs(opts)
    trainer = Trainer(opts)
    trainer.train()
