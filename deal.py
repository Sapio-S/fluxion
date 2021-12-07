import numpy as np
size=[10,25,50,100,200,300,400]
for ss in size:
    with open('log/1121/150_original_'+str(ss)) as f:
        text = f.read()
        sentence = text.split('\n')
        l = sentence[-3][1:-1]
        # print(l)
        li = l.split(',')
        lis = [float(x) for x in li]
        print(np.mean(lis))
        # print(lis)
        # break
        