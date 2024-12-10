import numpy as np
#import torch
import re
import random

PR = 5
a = np.zeros((114514,5))
b = np.zeros(114514)


N = 0

def readboard(filnm,Au,trainOrTest = 1,RR = 50,Ag = 0,Cu = 0):
    global a
    global b
    global N
    if(Ag == 0):
        Ag = 3*Au
    if(Cu == 0):
        Cu = 6*Au
    
    file = open(filnm,encoding = 'utf-8')
    s = file.readlines()
    n = len(s)
    nn = 0
    m = 0
    slv = np.zeros(1002)
    penal = np.zeros(1002)
    last_smt = np.zeros((1002,20)) #最后一次提交
    ap = np.zeros((1002,20))
    smt_tm = np.zeros((1002,20))
    ac = np.zeros((1002,20))
    smt_after_lock = np.zeros((1002,20))
    solve_time = np.zeros((1002,20))
    rks = np.zeros(1002)
    brk = np.zeros(1002)
    rk = 0
    for i in range(1,n):
        ss = s[i].split(' ')
        op = ss[0]
        t = ss[1]
        #print(op,t)
        if(op == '@problems'):
            m = int(t)    
        if(op == '@teams'):
            nn = int(t)    
        if(op == '@s'):
            tm,prb,st,wt,result = t.split(',')
            wt = int(wt)//60
            #print(wt)
            if(wt >= 240):
                R = i
                break
            x = int(tm)
            y = ord(prb)-ord('A')+1;
            #print(result)
            if(not ac[x][y]):
                last_smt[x][y] = wt
                smt_tm[x][y] += 1
                if(result == 'OK\n'):
                   # print(x,y)
                    ac[x][y] += 1

   # print(R)
                    
    dif = np.zeros(20)
    for i in range(1,nn+1):
        for j in range(1,m+1):
            if(ac[i][j]):
                dif[j] += 1
                slv[i] += 1
                penal[i] += 20*(smt_tm[i][j]-1)+last_smt[i][j]
        rks[i] = slv[i]*100000-penal[i]
        #if(i <= 100):
        #    print(i,rks[i])
    for j in range(1,m+1):
        dif[j] = Au/(dif[j]+1)
    # print(dif)
    for i in range(1,nn+1):
        cnt = 0
        for j in range(1,nn+1):
            if(rks[j] > rks[i]):
                brk[i] += 1
                cnt += 1
        brk[i] = (brk[i]+1)/Au
        
    for i in range(R+1,n):
        ss = s[i].split(' ')
        #print(ss)
        op = ss[0]
        t = ss[1]
        tm,prb,st,wt,result = t.split(',')
        wt = int(wt)//60
        x = int(tm)
        y = ord(prb)-ord('A')+1;
        #print(x,y)
        if(not ac[x][y]):
            smt_after_lock[x][y] = 1
            last_smt[x][y] = wt
            smt_tm[x][y] += 1
            for j in range(1,m+1):
                if(j != y and (last_smt[x][j] >= 240 or ac[x][j])):
                    solve_time[x][y] = max(last_smt[x][j],solve_time[x][y])
            if(trainOrTest and result == 'OK\n'):
                ac[x][y] = 1
              #  print(x,y)
    if(trainOrTest):
        for i in range(1,nn+1):
            for j in range(1,m+1):
                if(smt_after_lock[i][j]):
                    N += 1
                    a[N][0] = dif[j]
                    a[N][1] = last_smt[i][j]
                    a[N][2] = smt_tm[i][j]
                    a[N][3] = last_smt[i][j]-solve_time[i][j]
                    a[N][4] = brk[i]
                    if(ac[i][j]):
                        b[N] = 1
                    else:
                        b[N] = 0
                   # if(i <= 10):
                   #     print(N,a[N],b[N])
    else:
        for i in range(1,nn+1):
            for j in range(1,m+1):
                if(ac[i][j]):
                    ap[i][j] = RR
                if(smt_after_lock[i][j]):
                    sa = np.array([dif[j],last_smt[i][j],smt_tm[i][j],last_smt[i][j]-solve_time[i][j],brk[i]])
                    ap[i][j] = gP(sa,[1,2,1,1,1],RR)
        rnd_Ag = np.zeros(1002)
        rnd_Cu = np.zeros(1002)
        rnd_Au = np.zeros(1002)
                    #print(i,j,ap[i][j])
        for T in range(1,1001):
            rnd_rks = np.zeros(nn+1)
            for i in range(1,nn+1):
                for j in range(1,m+1):
                    x = random.randint(1,RR)
                    if(x <= ap[i][j]):
                        rnd_rks[i] += 100000-(20*(smt_tm[i][j]-1)+last_smt[i][j])
            rnd_rks = np.sort(rnd_rks)
            rnd_Au[T] = rnd_rks[nn-Au+1]
            rnd_Ag[T] = rnd_rks[nn-Ag+1]
            rnd_Cu[T] = rnd_rks[nn-Cu+1]
            
            #print(T,rnd_Au[T])
        rnd_Au = np.sort(rnd_Au)
        rnd_Ag = np.sort(rnd_Ag)
        rnd_Cu = np.sort(rnd_Cu)
        
        print('90% Au: problem solve',rnd_Au[900]//100000+1,'penalty:',100000-rnd_Au[900]%100000)
        print('70% Au: problem solve',rnd_Au[700]//100000+1,'penalty:',100000-rnd_Au[700]%100000)
        print('50% Au: problem solve',rnd_Au[500]//100000+1,'penalty:',100000-rnd_Au[500]%100000)
        print('30% Au: problem solve',rnd_Au[300]//100000+1,'penalty:',100000-rnd_Au[300]%100000)
        print('10% Au: problem solve',rnd_Au[100]//100000+1,'penalty:',100000-rnd_Au[100]%100000)
        
        
        print('90% Ag: problem solve',rnd_Ag[900]//100000+1,'penalty:',100000-rnd_Ag[900]%100000)
        print('70% Ag: problem solve',rnd_Ag[700]//100000+1,'penalty:',100000-rnd_Ag[700]%100000)
        print('50% Ag: problem solve',rnd_Ag[500]//100000+1,'penalty:',100000-rnd_Ag[500]%100000)
        print('30% Ag: problem solve',rnd_Ag[300]//100000+1,'penalty:',100000-rnd_Ag[300]%100000)
        print('10% Ag: problem solve',rnd_Ag[100]//100000+1,'penalty:',100000-rnd_Ag[100]%100000)

        print('90% Cu: problem solve',rnd_Cu[900]//100000+1,'penalty:',100000-rnd_Cu[900]%100000)
        print('70% Cu: problem solve',rnd_Cu[700]//100000+1,'penalty:',100000-rnd_Cu[700]%100000)
        print('50% Cu: problem solve',rnd_Cu[500]//100000+1,'penalty:',100000-rnd_Cu[500]%100000)
        print('30% Cu: problem solve',rnd_Cu[300]//100000+1,'penalty:',100000-rnd_Cu[300]%100000)
        print('10% Cu: problem solve',rnd_Cu[100]//100000+1,'penalty:',100000-rnd_Cu[100]%100000)

        
N = 0
def getboard():
    file = open('trainboard.txt',encoding = 'utf-8')
    s = file.readlines()
    global N
    N = 0
    for t in s:
        tt = t.split()
        bd = tt[0]
        aun = int(tt[1])
        readboard(bd,aun)
getboard();

d = np.zeros((114514,5))
ave = np.zeros(6)
var = np.zeros(6)
for j in range(0,5):
    for i in range(1,N+1):
        ave[j] += a[i][j]
    ave[j] /=  N
    for i in range(1,N+1):
        var[j] += (a[i][j]-ave[j]) ** 2
    var[j] = var[j] ** 0.5
    for i in range(1,N+1):
        d[i][j] = (a[i][j]-ave[j])/var[j]

def gP(sa,w,R = 20):
    p = []
    for i in range(0,5):
        sa[i] = (sa[i]-ave[i])/var[i]
    for j in range(1,N+1):
        dis = 0 
        for k in range(0,5):
            dis += (w[k]*(sa[k]-d[j][k]) ** 2)
        p.append([dis,b[j]])
        #print(j,dis)
    #print(p)
    sp = np.array(p)
    spidx = np.argsort(sp[:, 0])
    ff = sp[spidx]
    cnt = 0
    for k in range(0,R):
        cnt += ff[k][1]
    return cnt

while(1):
    contest,Aum,Agm,Cum = input().split()
    Aum = int(Aum);
    Agm = int(Agm);
    Cum = int(Cum);
    readboard(contest,Aum,0,20,Agm,Cum)
