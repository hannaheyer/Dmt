# Simple script that returns the best feature set.
import numpy

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-filename')
    args = parser.parse_args()

    f = open(args.filename,'r')
    lines = f.readlines()
    f.close()
    best_v = 0
    best_f = []
    best_i = -1
    for line in lines:
    	sp = line.split('|')
    	v = float(sp[2][1:])
    	i = (sp[0])
    	feats = sp[3].split('min')[1].strip()
    	if v > best_v:
    		best_v = v
    		best_f = feats
    		best_i = i
    print(best_i,':',best_v,':',str(best_f))
		
