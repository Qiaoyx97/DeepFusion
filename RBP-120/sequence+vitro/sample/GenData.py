from glob import glob

name_list = glob('*')

for name in name_list:
    if '.py' in name:
        continue
    out_pos = open(name+'/'+name+'.seq.pos.txt', 'w')
    out_neg = open(name+'/'+name+'.seq.neg.txt', 'w')
    with open( name+'/'+name+'.train.seq.txt') as f:
        for line in f.readlines():
            if line.split('\t')[0] == '1':
                print(line.strip('\n'), file = out_pos)
            else:
                print(line.strip('\n'), file = out_neg)
    
    out_vitropos = open(name+'/'+name+'.vitro.pos.txt', 'w')
    out_vitroneg = open(name+'/'+name+'.vitro.neg.txt', 'w')
    with open( name+'/'+name+'.train.vitro.txt') as f:
        for line in f.readlines():
            if line.split('\t')[0] == '1':
                print(line.strip('\n'), file = out_vitropos)
            else:
                print(line.strip('\n'), file = out_vitroneg)