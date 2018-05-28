import time

# Used to add property id if that was forgotten during test file generation
f = open('test/full_test.txt','r')
out = open('test/full_testEdit.txt','w')
def get_prop_id(line):
    vals = line.split(' ')[1:]
    prop_id = [x[1] for x in [y.split(':') for y in vals] if x[0] == '7']
    return prop_id
s = time.time()
lines = f.readlines()
f.close()
n = len(lines)
part_i = n/10
print('Reading input lines took {} mins'.format((time.time()-s)/60)) 
for i,line in enumerate(lines):
    p_id = get_prop_id(line)[0].split("'")[0]
    #print(p_id)
    out.write(line.strip()+'#p_id:{}\n'.format(p_id))
    if i%100000 == 0:
        print('{}/{}'.format(i, n))
        print(p_id)
out.close()
print('Whole thing took {} minutes'.format((time.time()-s)/60))