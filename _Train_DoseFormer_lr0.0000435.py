import os
import torch
name = '_Train_DoseFormer_lr0.0000435_lite_retrain'
if not os.path.exists('Model_Log/{}'.format(name)):
    os.mkdir('Model_Log/{}'.format(name))
    print("create dir " + 'Model_Log/{}'.format(name))
if not os.path.exists('Model_Save/{}'.format(name)):
    os.mkdir('Model_Save/{}'.format(name))
    print("create dir " + 'Model_Save/{}'.format(name))


from utils.utils import data_v
from Model_DoseFormer.DoseFormer_v2_lite import cnn_lstm_attention_gt
import datetime



allseq2t,allst,idx_train,idx_val,idx_test,y,idx_train_dynamic,idx_val_dynamic,idx_test_dynamic,idx_train_static,idx_val_static,idx_test_static = data_v(device='cuda:0')



import torch.nn.functional as F

from utils.utils import accuracy, test_para

torch.cuda.manual_seed(3407)
torch.manual_seed(3018)
LR = 0.000435
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tg = cnn_lstm_attention_gt()
tg.float()
tg.cuda(device=0)
loss_function = F.nll_loss
optimizer = torch.optim.Adam(tg.parameters(), lr=LR)

epochs = 2000


model_savepath = "Model_Save/"+name+"/model_epo_{}_loss_{}_acc_{}.pkl"
train_log_path = "Model_Log/"+name+"/train_log.txt"
val_log_path = "Model_Log/"+name+"/val_log.txt"
test_log_path = "Model_Log/"+name+"/test_log.txt"
starttime = datetime.datetime.now()




for epoch in range(epochs):
    
    tg.train()
    idx = idx_train
    idx_dynamic = idx_train_dynamic
    idx_static = idx_train_static
    out ,adj= tg(allseq2t[idx_dynamic],allst[idx_static])
    optimizer.zero_grad()
    loss = loss_function(out[idx],y[idx])
    loss.backward()
    optimizer.step()
    acc = accuracy(out[idx],y[idx])
    accp , lr = test_para(out,y,idx)

    sen = lr[4]
    spc = lr[5]
    losst = float(loss)
    acct = float(acc)

    if epoch % 10 ==0:
        endtime = datetime.datetime.now()
        print("epoch : {} , loss : {} , acc : {} , time : {}s".format(
            epoch,
            float(loss),
            float(acc),
            (endtime - starttime).seconds
        ))
        print(lr)
        starttime = datetime.datetime.now()
    if epoch % 100 ==0:
        with open(train_log_path,'a+') as f:
            print(
                epoch,
                float(loss),
                float(acc),
                lr,
                file=f
            )
    if epoch % 100 ==0:
        if float(acc) >= 0.8:
            torch.save(tg, model_savepath.format(
                                                epoch,
                                                float(loss),
                                                float(acc)
                                                )
                                                )
             
    if epoch % 10 == 0:
        tg.eval()
        with open(test_log_path,'a+') as f:
            idx = idx_test
            idx_dynamic = idx_test_dynamic
            idx_static = idx_test_static
            out ,adj= tg(allseq2t[idx_dynamic],allst[idx_static])
            loss = loss_function(out[idx],y[idx])
            acc = accuracy(out[idx],y[idx])
            accp , lr = test_para(out,y,idx)
            print(
                epoch,
                float(loss),
                float(acc),
                lr,
                file=f
            )
        with open(val_log_path,'a+') as f:
            idx = idx_val
            idx_dynamic = idx_val_dynamic
            idx_static = idx_val_static
            out ,adj= tg(allseq2t[idx_dynamic],allst[idx_static])
            loss = loss_function(out[idx],y[idx])
            acc = accuracy(out[idx],y[idx])
            accp , lr = test_para(out,y,idx)
            print(
                epoch,
                float(loss),
                float(acc),
                lr,
                file=f
            )

                                    
        



    


    

    