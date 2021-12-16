import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default="", help='config path')

parser.add_argument("--reference", type=str, default="", help='path to reference heightfield')


args = parser.parse_args()

for arg in vars(args):
     if vars(args)[arg] == 'True':
          vars(args)[arg] = True
     elif vars(args)[arg] == 'False':
          vars(args)[arg] = False
