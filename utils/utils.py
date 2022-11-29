import csv


global_device = 'cpu'



def load_static_data(path="data/static/X_static_select.csv"):
    sid =[]
    cols = []
    data_static = []
    labels = []
    with open(path,'r',encoding="utf-8") as f:  
        lines=csv.reader(f)
        k = 1
        for line in lines:
            if k == 1:
                cols = line[1:-2]
                k +=1
                continue
            sid.append(line[0])
            data_static.append(line[1:-1])
            labels.append(line[-1])
    return cols,sid,data_static,labels



import time
def makestamp(stringtime):
    if ":" not in stringtime:
        stringtime = stringtime + " 00:00"
    timeArray = time.strptime(stringtime, "%Y/%m/%d %H:%M")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def naz(l): # line has 0 value is ZERO
    for i in l:
        if i==0:
            return False
    return True

def getphsic(path,encoding='utf-8'):
    physic = []
    with open(path,'r',encoding=encoding) as f:  
            lines=csv.reader(f)
            k = 1
            for line in lines:
                if k == 1:
                    k+=1
                    continue
                physic.append(line)
    phycols = [
    'SHOUSHUID', 
    '手术开始时间', 
    '病历号',
    '姓名',
    '数据采集时间',
    '收缩压',
    '舒张压',
    '动脉收缩压',
    '动脉舒张压', 
    '心率',
    '脉搏',
    '饱和度'
    ]
    physict = []

    for i in physic:
        t = []
        for j in range(1,5):
            
            t.append(i[j])
        t.append(makestamp(i[5])-makestamp(i[2]))
        for j in range(6,len(i)):
            
            t.append(float(i[j]))
        physict.append(t)

    print("physic data length for {}:".format(path),len(physict))

    physictcol = [
        'sID',
        '手术开始时间', 
        '病历号',
        '姓名',
        'timeDuring',
        '收缩压',
        '舒张压',
        '动脉收缩压',
        '动脉舒张压', 
        '心率',
        '脉搏',
        '饱和度'

    ]
    return physictcol,physict







def load_dynamic_data(paths = [
                        'data/dynamic/2018tz.csv',
                        'data/dynamic/2019tz.csv',
                        'data/dynamic/2020tz.csv'
                    ],
                      Select = True, # choose all data or data sid in static data
                      static_path = "data/static/X_static_select.csv",
                      cut = 10 # to select the sid whose zero line for head is less than or equal to cutnumber
                    ): 
    

    phycols , physic2018 = getphsic(paths[0],encoding='gbk')
    _ , physic2019 = getphsic(paths[1],encoding='gbk')
    _ , physic2020 = getphsic(paths[2],encoding='gbk')

    physic = []
    for i in physic2018:
        physic.append(i)
    for i in physic2019:
        physic.append(i)
    for i in physic2020:
        physic.append(i)
    del physic2018
    del physic2019
    del physic2020

    physel = []
    
    physel = physic
        
    datad = {}
    for i in physel:
        if i[0] not in datad:
            datad[i[0]] = {}
        t = []
        for j in range(5,12):
            t.append(i[j])
        datad[i[0]][i[4]] = t
    s = 0 

    for i in datad:    
        t1 = {}
        t2 = datad[i]
        for j in sorted(t2):
            t1[j] = datad[i][j]
        datad[i] = t1


    datanz = {} # from k'st line is not all zero
    for i in datad:
        datanz[i] = 0 
        for j in datad[i]:
            if not naz(datad[i][j]):
                datanz[i] +=1
            else:
                break
    cutd = {}
    for i in datad:
        if datanz[i] <= cut:
            cutd[i] = 0
    
    datadc = {}
    for i in datad:
        if i in cutd:
            datadc[i] = datad[i]


    # add value to the zero , strategy : if it is the zero for head , find the neraest value , if it is in the seq , find the last value
    for i in datadc:
        for j in datadc[i]:
            for k in range(7):
                tseq1 = []
                if datadc[i][j][k] == 0:
                    
                    for p in datadc[i]:
                        if p <= j and datadc[i][p][k] !=0:
                            tseq1.append(datadc[i][p][k])
                        if p>j:
                            break
                tseq1.reverse()
                if len(tseq1) != 0:
                    datadc[i][j][k] = tseq1[0]
                else:
                    for p in datadc[i]:
                        if p  > j and datadc[i][p][k] !=0:
                            
                            datadc[i][j][k]=datadc[i][p][k]
                            break
    dataseq = []
    for i in datadc:
        line = []
        line.append(i)
        t = []

        for j in range(7):
            t.append([])

        cur = None

        flag = True

        ifbreak = False
        
        for j in datadc[i]:

            ifcontinue = False
            if flag:
                cur = datadc[i][j]
                # print(cur)
                flag = False

            for k in range(7):
                if datadc[i][j][k] < cur[k] / 2:
                    ifbreak = True
                    break
                if datadc[i][j][k] <= 10:
                    ifcontinue = True
                    break
            
            cur = datadc[i][j]
            if ifbreak:
                break
            if ifcontinue:
                continue
            for k in range(7):
                t[k].append(datadc[i][j][k])
        for j in t:
            line.append(j)
        dataseq.append(line)
    
    cols = ['ID',
            '收缩压',
            '舒张压',
            '动脉收缩压',
            '动脉舒张压',
            '心率',
            '脉搏',
            '饱和度']

    dataseqsel = []
    if Select:
        
        _,sid,_,_= load_static_data(path=static_path)
       
        for i in sid:
            for j in dataseq:
                if i == j[0]:
                    dataseqsel.append(j)

        dataseq = dataseqsel
    return dataseq,cols

def draw1pic(y,sid,name,
            save = True,
            show = True,
            savepath = 'pic/datashow/'): # l is data sequence , cols is names,s is start , e is end):
    import matplotlib.pyplot as plt 
    if save and not show:
        import matplotlib
        matplotlib.use("Agg")

    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.rcParams['axes.unicode_minus']=False 
    plt.figure(figsize=(6,6), dpi=80)
    x = range(len(y))
    plt.plot(x, y, marker='.', mec='r', mfc='w',label=name)
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.title(str(sid)+"/"+name) #标题
    if save:
        import os
        spath  = savepath + str(sid)
        if not os.path.exists(spath):
            os.mkdir(spath)

        plt.savefig(spath+"/{}.png".format(name))
    if show:
        plt.show()
    plt.close('all')

def drawpic(l,cols,s,e,
            save = True,
            show = True,
            savepath = 'pic/datashow/'): # l is data sequence , cols is names,s is start , e is end
    print("create sid {} pic from {} to {} ".format(l[0],s,e))
    for i in range(s,e+1):
        draw1pic(l[i],l[0],cols[i],save,show,savepath)

import numpy as np

def norm(data,t):
    res = []
    
    for i in data:
        tmp = np.zeros((7,len(i[1])))
        for j in range(1,8):
            for k in range(len(i[j])):
                #  TODO CHANGE
                tmp[j-1][k] = (i[j][k] - t[j-1][1]) / (t[j-1][0] -t[j-1][1])
        res.append(np.array(tmp))
    return res

def load_dynamic_model_data(paths = [
                        'data/dynamic/2018tz.csv',
                        'data/dynamic/2019tz.csv',
                        'data/dynamic/2020tz.csv'
                    ],
                      Select = True, # choose all data or data sid in static data
                      static_path = "data/static/X_static_select.csv",
                      cut = 10, # to select the sid whose zero line for head is less than or equal to cutnumber
                    ):
    cols,sid,datastatic,labels= load_static_data(path=static_path)
    dataseq,cols2=  load_dynamic_data(paths,Select,static_path,cut)
    traincut = 300
    vaildcut = 400

    testcut = 510

    labels_i = []
    for i in labels:
        labels_i.append(int(i))
    Y = np.array(labels_i)

    t = np.zeros((7,4))

    for i in range(7):
        t[i][1] = 0x7fff



    for i in dataseq[0:traincut]:

        for j in range(1,8):

            for k in i[j]:
            
                t[j-1][0] = max(t[j-1][0],k)
                t[j-1][1] = min(t[j-1][1],k)
                t[j-1][2] += k 
                t[j-1][3] += 1

    for i in range(7):
        t[i][2] = t[i][2]/t[i][3]

    X_train =  norm(dataseq[0:traincut],t)
    y_train = Y[0:traincut]

    X_val = norm(dataseq[traincut+1:vaildcut],t)
    y_val = Y[traincut+1:vaildcut]

    X_test = norm(dataseq[vaildcut+1:-1],t)
    y_test = Y[vaildcut+1:-1]

    return X_train,y_train,X_val,y_val,X_test,y_test
import torch
def pad_mask(x,pad_size = 64):
    x = x.T
    mask = np.zeros(pad_size)
    embed_size = x.shape[1]
    if x.shape[0] <= pad_size:
        mask[x.shape[0]:] = 1

        x = np.pad(x,((0,pad_size-x.shape[0]),(0,0)),'constant',constant_values = (0,0))
    else:
        x = x[-pad_size:,:]
        mask[:] = 0
    x = torch.from_numpy(x)
    mask = torch.from_numpy(mask)
    return x.view(1,pad_size,embed_size).to(torch.float32),mask.to(torch.float32)

def pad_batch(X,pad_size= 64,embed_size = 7):
    xt = np.zeros((len(X),pad_size,embed_size))
    m = np.zeros((len(X),pad_size))
    for i in range(len(X)):
        t,mask = pad_mask(X[i])
        xt[i] = t
        m[i] = mask
    xt = torch.from_numpy(xt)
    m = torch.from_numpy(m)
    # print(xt.shape)
    # print(m.shape)
    return xt.to(torch.float32),m.to(torch.float32)

def _train_batch_(X,y,pad_size= 64,embed_size = 7,batch_size = 32):
    xt = np.zeros((batch_size,pad_size,embed_size))
    yt = np.zeros(batch_size)
    m = np.zeros((batch_size,pad_size))


    n = len(y)
    sumt = sum(y)
    
    
    lp = []
    ln = []

    for i in range(n):
        if y[i] == 0:
            ln.append(i)
        else:
            lp.append(i)

    batch_no = np.zeros(batch_size)

    lp = np.array(lp)
    ln = np.array(ln)

    np.random.shuffle(lp)
    np.random.shuffle(ln)

    for i in range(batch_size//2):
        batch_no[i] = lp[i]
        batch_no[i+batch_size//2] = ln[i]
    
    np.random.shuffle(batch_no)
    
    bt = 0
    for j in batch_no:
        i = int(j)
        t,mask = pad_mask(X[i])
        xt[bt] = t
        m[bt] = mask
        yt[bt] = y[i]
        bt += 1
    xt = torch.from_numpy(xt)
    m = torch.from_numpy(m)
    yt = torch.from_numpy(yt)
    # print(xt.shape)
    # print(m.shape)
    return xt.to(torch.float32),m.to(torch.float32),yt.long()

def accuracy(yp,yt):
    ypl = torch.argmax(yp,dim=1)
    acc = 0.0
    n = yt.shape[0]
    for i in range(n):
        if ypl[i] == yt[i]:
            acc += 1
    acc /= n
    return acc

def sensitive(yp,yt):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ypl = torch.argmax(yp,dim=1)
    n = yt.shape[0]
    for i in range(n):
        if ypl[i] == 1 and yt[i] == 1:
            TP += 1
        if ypl[i] == 1 and yt[i] == 0:
            FP += 1
        if ypl[i] == 0 and yt[i] == 1:
            FN += 1
        if ypl[i] == 0 and yt[i] == 0:
            TN += 1
        
    return TP/(TP+FN)



def test(m,xt,yt):
    m.eval()
    valacc = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for x,y in zip(xt,yt):
        x,mask = pad_mask(x)
        out =m(x,mask)
        if torch.argmax(out, dim=1) == y:
            valacc += 1
        if torch.argmax(out,dim=1) == 1 and y == 1:
            TP += 1
        if torch.argmax(out,dim=1) == 1 and y == 0:
            FP += 1
        if torch.argmax(out,dim=1) == 0 and y == 0:
            TN += 1
        if torch.argmax(out,dim=1) == 0 and y == 1:
            FN += 1
    valacc /= len(xt)
    
    return valacc , [TP,TN,FP,FN]


def test_para(out,y,idx):
    def pyacc(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)
    valacc = pyacc(out[idx],y[idx])
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    yp = torch.argmax(out[idx],dim=1)

    for i,j in zip(yp,y[idx]):
        if i == 1 and j == 1:
            TP+=1
        if i == 0 and j == 1:
            FP+=1
        if i == 0 and j == 0:
            TN += 1
        if i == 1 and j == 0:
            FN += 1

    if TP == 0 or TP+FN == 0 :
        sen = 0
    else:
        sen = TP/(TP+FN)
    if TN == 0 or TN+FP == 0 :
        spe = 0
    else:
        spe = TN/(TN+FP)
    
    return valacc , [TP,TN,FP,FN,sen,spe] 
def test_reg(m,xt,yt):
    m.eval()
    
    valaccl = []
    valda = {}
    for i in range(-10,11):
        thi = i/10.0
        valda[str(thi)] = {}
        valacc = 0
        vald = {}
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for x,y in zip(xt,yt):
            x,mask = pad_mask(x)
            out =m(x,mask)
            out = out.view(-1)
            
            if out >= thi and y == 1:
                TP += 1
            if out >= thi and y == 0:
                FP += 1
            if out < thi and y == 0:
                TN += 1
            if out < thi and y == 1:
                FN += 1
        valda[str(thi)]["ACC"] = (TP+TN) / (len(xt)+1e-7)
        valda[str(thi)]["PRECESION"] = TP / ((TP+FP)+1e-7)
        valda[str(thi)]["SENTITIVE"] = TP / ((TP+FN)+1e-7)
        valda[str(thi)]["TP"] = TP
        valda[str(thi)]["TN"] = TN
        valda[str(thi)]["FP"] = FP
        valda[str(thi)]["FN"] = FN
        
        
        
    
    return valda

def process_predata(dataseq,srclen=30,trglen=6,device=global_device):
    res = []
    for i in dataseq:
        if len(i[1]) < trglen * 2:
            continue
        t = i[1:]
        t = np.array(t)
        t = t.T
        lt , _ = t.shape
        trg = t[lt-trglen:]
        if lt - trglen - srclen-1 >=0:
            src = t[lt-trglen-srclen:lt-trglen]
        else:
            src = t[:lt-trglen]
            src = np.pad(src, [(srclen - src.shape[0], 0), (0, 0)], mode="edge")
        trg = torch.from_numpy(trg).to(torch.float32).to(device)
        src = torch.from_numpy(src).to(torch.float32).to(device)
        res.append((src,trg))
    return res

def process_traindata(dataseq,srclen=30,device=global_device):
    res = []
    resall = []
    for i in dataseq:
        t = i[1:]
        t = np.array(t)
        t = t.T
        lt , _ = t.shape
        if lt-srclen >=0:
            src = t[lt-srclen:]
        else:
            src = t
            src = np.pad(src, [(srclen - src.shape[0], 0), (0, 0)], mode="edge")
        resall.append(src)
        src = torch.from_numpy(src).to(torch.float32).to(device).view(1,srclen,-1)
        res.append(src)
    
    return res

        
        

    


def get_pretrain_data(Select = False):
    dataseq,cols = load_dynamic_data(Select=Select)
    traincut = int(len(dataseq) * 0.9)
    valcut = int(len(dataseq) * 0.1)
    testcut = int(len(dataseq) * 0.1)

    trainseq = dataseq[0:traincut]
    valseq = dataseq[traincut+1:traincut+valcut]
    testseq = dataseq[traincut+valcut+1:]

    train = process_predata(trainseq)
    val = process_predata(valseq)
    test = process_predata(testseq)
    return train,val,test



def process_static(datastatic,device=global_device,datatype=torch.long):
    res = []
    for i in datastatic:
        t = []
        for j in i:
            t.append(float(j))
        t = np.array(t)
        t = torch.from_numpy(t).to(datatype).to(device)
        res.append(t)
    return res

def t2l(data):
    a = []
    for i in data:
        a.append(i.to(global_device).numpy().tolist())
    return a
from sklearn.preprocessing import OneHotEncoder

def get_train_data():
    dataseq,cols = load_dynamic_data(Select=True) # 510 
    traincut = int(len(dataseq) * 0.8)
    valcut = int(len(dataseq) * 0.1)
    testcut = int(len(dataseq) * 0.1)
    
    trainseq = dataseq[0:traincut]
    valseq = dataseq[traincut+1:traincut+valcut]
    testseq = dataseq[traincut+valcut+1:]

    trainseq = process_traindata(trainseq)
    valseq = process_traindata(valseq)
    testseq = process_traindata(testseq)

    # all
    allseq = t2l(process_traindata(dataseq))
    


    st_cols,sid,datastatic,labels= load_static_data(path="data/static/X_static_select.csv")  # 28 * 510

    numl = [
        4,5,24,25,26,27
    ]
    enc = OneHotEncoder(handle_unknown='ignore')
    static_no_num = []
    static_num = []

    for i in datastatic:
        t1 = []
        t2 = []
        for j in range(28):
            if j in numl:
                t1.append(i[j])
            else:
                t2.append(i[j])
        static_num.append(t1)
        static_no_num.append(t2)
    enc.fit(static_no_num)
    static_no_num = enc.transform(static_no_num).toarray()


    
                

    labels_i = []
    for i in labels:
        labels_i.append(int(i))
    Y = np.array(labels_i)

    static_no_num_train = static_no_num[0:traincut]
    static_no_num_val = static_no_num[traincut+1:traincut+valcut]
    static_no_num_test = static_no_num[traincut+valcut+1:]

    static_no_num_train = process_static(static_no_num_train,datatype=torch.long)
    static_no_num_val = process_static(static_no_num_val,datatype=torch.long)
    static_no_num_test = process_static(static_no_num_test,datatype=torch.long)

    # all
    all_static_no_num = t2l(process_static(static_no_num,datatype=torch.long))

    static_num_train = static_num[0:traincut]
    static_num_val = static_num[traincut+1:traincut+valcut]
    static_num_test = static_num[traincut+valcut+1:]

    static_num_train = process_static(static_num_train,datatype=torch.float32)
    static_num_val = process_static(static_num_val,datatype=torch.float32)
    static_num_test = process_static(static_num_test,datatype=torch.float32)

    # all
    all_static_num = t2l(process_static(static_num,datatype=torch.long))


    y_train = Y[0:traincut].T
    y_val = Y[traincut+1:traincut+valcut].T
    y_test = Y[traincut+valcut+1:].T

    traincut = 407
    valcut = 51
    testcut = 51
    
    idx_train = range(0,traincut)
    idx_val = range(traincut,traincut+valcut)
    idx_test = range(traincut+valcut,len(Y))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    yall = torch.from_numpy(Y).long()

    return trainseq,valseq,testseq,\
           static_no_num_train,static_no_num_val,static_no_num_test,  \
           static_num_train,static_num_val,static_num_test, \
           y_train,y_val,y_test , \
           allseq,all_static_no_num,all_static_num, \
           idx_train,idx_val,idx_test ,yall



def get_data_DynamicStatic(device='cpu'):
    
    import torch
    seq , col = load_dynamic_data()
    import numpy as np
    cnn_inputs = []
    for i in seq:
        state = np.array(i[1:])
        slices = np.split(state, list(range(5, state.shape[1]-5, 5)), axis=1)
        cnns = [torch.from_numpy(np.expand_dims(s, axis=0).copy()).float().to(device) for s in slices]
        cnn_inputs.append(cnns)
    from utils.utils import get_train_data
    trainseq,valseq,testseq,\
    static_no_num_train,static_no_num_val,static_no_num_test,  \
    static_num_train,static_num_val,static_num_test, \
    y_train,y_val,y_test , \
    allseq,all_static_no_num,all_static_num, \
    idx_train,idx_val,idx_test,y = get_train_data()
    allstatic = []
    for i,j in  zip(all_static_no_num,all_static_num):
        t = i+j
        allstatic.append(t)


    static_tensors = []
    for i in allstatic:
        static_tensors.append(torch.from_numpy(np.array(i)).to(device).to(torch.float32).view(1,1,129)) 

    allseq2 = []
    for i in range(509):
        t = []
        t1 = allseq[i]
        for j in range(6):
            allseq2.append(t1[0][j*5:(j+1)*(5)])
    allseq2t = torch.from_numpy(np.array(allseq2)).view(-1,7,5).to(torch.float32).to(device)

    allst = np.array(allstatic)
    allst = torch.from_numpy(allst).to(torch.float32).to(device)
    y = y.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    return allseq2t,allst,idx_train,idx_val,idx_test,y

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def data_v(device='cpu'):
    allseq2t,allst,idx_train,idx_val,idx_test,y = get_data_DynamicStatic(device=device)
    N = 509
    traincut = 407
    valcut = 51
    testcut = 51
    
    
    idx_train_static = range(0,traincut)
    idx_val_static = range(0,traincut+valcut)
    idx_test_static = range(0,traincut+valcut+testcut)

    idx_train_static = torch.LongTensor(idx_train_static).to(device)
    idx_val_static = torch.LongTensor(idx_val_static).to(device)
    idx_test_static = torch.LongTensor(idx_test_static).to(device)

    idx_train_dynamic = range(0,6 * traincut)
    idx_val_dynamic = range(0,6 * (traincut+valcut))
    idx_test_dynamic = range(0,6 * (traincut+valcut+testcut))
    idx_train_dynamic = torch.LongTensor(idx_train_dynamic).to(device)
    idx_val_dynamic = torch.LongTensor(idx_val_dynamic).to(device)
    idx_test_dynamic = torch.LongTensor(idx_test_dynamic).to(device)
    return allseq2t,allst,idx_train,idx_val,idx_test,y,idx_train_dynamic,idx_val_dynamic,idx_test_dynamic,idx_train_static,idx_val_static,idx_test_static



if __name__ == "__main__":
    
    get_train_data()

        

        
        
        
    