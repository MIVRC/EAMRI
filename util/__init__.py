from .imageUtil import *
from .recoder import *

def paramNumber(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        #print('ddd: '+str(list(i.size())))
        l = 1
        for j in i.size():
            l *= j
        #print('aaa: '+ str(l))
        k = k + l
    
    return k



