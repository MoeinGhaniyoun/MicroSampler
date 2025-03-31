import re, sys, pickle

appname = sys.argv[1]
keyname = sys.argv[2]
uarch = sys.argv[3]
suite = sys.argv[4]

#Not all tests print key bits during loop for efficiency, use the report from debug runs
simout = open('logs/'+uarch+'/'+suite+'/'+appname+'/100/'+keyname+'/stdout.txt', 'r').read() 
loopsUArch, theta_lst, diff, keybin = pickle.load(open('logs/'+uarch+'/'+suite+'/'+appname+'/100/'+keyname+'/sets.pickle', 'rb'))

keybin = ''.join(keybin)
key = ''
for match in re.finditer('====== Round [0-9]+, e=([0|1])  ======', simout):
    key += match.group(1)

print(key)
print(keybin)

assert key == keybin, 'FAIL: key mismatch for '+keyname
