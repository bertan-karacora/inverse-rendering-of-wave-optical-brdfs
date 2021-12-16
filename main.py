from options import args
import yaml
import genBrdf

if args.config != "":
    with open(args.config, "r") as f:
        config = yaml.load(f)
    for op in config:
        setattr(args, op, config[op])

arr = genBrdf.genBrdf()
print(type(arr))
print(arr[1, 0, 2])
