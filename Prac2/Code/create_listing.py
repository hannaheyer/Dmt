from collections import defaultdict
import operator
import time
import csv

# Converts model scores into a ranking of 

def parse_line(line):
    l = line.split(' ')
    p_id = l[2].split(':')[1] 
    assert l[2].split(':')[0] == 'p_id'
    q_id = l[0]
    score = l[4]
    line = '{},{}\n'.format(q_id,p_id)
    return (int(q_id), p_id, float(score))


def write_dict(s_dict,score_file):
    score_id = score_file.split('.')[0]
    listing = open('{}_listing.csv'.format(score_id),'w')
    writer = csv.writer(listing, delimiter=',')
    writer.writerow(['SearchId', 'PropertyId'])
    m = len(s_dict.keys())
    for i,q_id in enumerate(sorted(s_dict.keys())):
        sorted_x = sorted(s_dict[q_id].items(), key=operator.itemgetter(1), reverse=True)
        for (k,v) in sorted_x:
            if q_id % 10 == 0:
                print(q_id,k,v)
            #listing.write('{},{}\n'.format(q_id,k))
            writer.writerow([q_id,k])
        if i%10000 == 0:
            print('{}/{}'.format(i, m))
    listing.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-score',type=int)
    args = parser.parse_args()

    s = time.time()
    print("Starting listing creation.")
    score_file = open(args.score,'r')
    lines = score_file.readlines()
    score_file.close()
    score_dict =defaultdict(dict)
    n = len(lines)
    print('Parsing score file..')
    for i,line in enumerate(lines):
        (q_id, p_id, score) = parse_line(line)
        score_dict[q_id][p_id] = score
        if i%100000 == 0:
            print('{}/{}'.format(i, n))

    print('Writing listing')
    write_dict(score_dict, args.score)

    print("DONE writing listing in {}secs".format(time.time()-s))

