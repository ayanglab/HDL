import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from visdom import Visdom
from torch.utils.data import Dataset, DataLoader
from biDataset import biDataset
import Models
from torch.autograd import Variable
from losses_pytorch.boundary_loss import SoftDiceLoss

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'
gpu = 1
loss_time = torch.nn.CrossEntropyLoss()
loss_seg_dc = SoftDiceLoss()
loss_func = torch.nn.L1Loss(reduction='mean')
loss_func_mask = torch.nn.L1Loss(reduction='sum')

def trainEpoch(epoch,train_loader,test_loader,net,optimizer,loss_func):
    train_loss = 0
    train_loss_l1 = 0
    train_loss_seg = 0
    train_loss_l1_w1 = 0
    test_loss = 0
    test_loss_l1 = 0
    test_loss_seg = 0
    test_loss_l1_w1 = 0
    for step, data in enumerate(train_loader):
        ###get data from dict
        img =  Variable(data['image'],requires_grad=True).cuda().float()
        label = Variable(data['label'],requires_grad=True).cuda().float()
        seg = Variable(data['seg'],requires_grad=True).cuda().float()
        timeGap =  Variable(data['timeGap'],requires_grad=True).cuda().float()
        weightmap1 = data['W1'].cuda().float()
        weightmap2 = data['W2'].cuda().float()
        
        net = net.train()
        img_output,seg_output = net(img,seg,timeGap)
        seg_map = label[:,1,:,:].unsqueeze(1)
        l1los_all = loss_func(img_output, label[:, 0, :, :].unsqueeze(1))
        l1los_w1 = (loss_func_mask(img_output * weightmap1.unsqueeze(1),
                                label[:, 0, :, :].unsqueeze(1) * weightmap1.unsqueeze(1)))/torch.sum(weightmap1)
        l1los_w2 = (loss_func_mask(img_output * weightmap2.unsqueeze(1),
                                label[:, 0, :, :].unsqueeze(1) * weightmap2.unsqueeze(1)))/torch.sum(weightmap2)
        los_seg = loss_seg_dc(seg_output, seg_map)
        tr_los = l1los_all + l1los_w1 + l1los_w2 + los_seg
        optimizer.zero_grad()
        tr_los.backward()
        optimizer.step()

        train_loss+=float(tr_los.data)
        train_loss_l1 += float(l1los_all.data)
        train_loss_seg += float(los_seg.data)

        train_loss_l1_w1 += float(l1los_w1.data)
        if step%50 ==0:
            print('Epoch:', epoch, '|Step:', step,
              '|train loss:', tr_los.data,'|l1 loss:', l1los_all.data,
                  '|l1 loss on roi:', l1los_w1.data,'|segmentation loss:', los_seg.data)

    with torch.no_grad():
        net = net.eval()
        for i, data in enumerate(test_loader):
            img = Variable(data['image'], requires_grad=True).cuda().float()
            label = Variable(data['label'], requires_grad=True).cuda().float()
            timeGap = Variable(data['timeGap'], requires_grad=True).cuda().float()
            weightmap1 = data['W1'].cuda().float()
            weightmap2 = data['W2'].cuda().float()

            img_output, seg_output = net(img,seg,timeGap)
            seg_map = label[:, 1, :, :].unsqueeze(1)
            l1los_all = loss_func(img_output, label[:, 0, :, :].unsqueeze(1))
            l1los_w1 = (loss_func_mask(img_output * weightmap1.unsqueeze(1),
                                       label[:, 0, :, :].unsqueeze(1) * weightmap1.unsqueeze(1))) / torch.sum(
                weightmap1)
            l1los_w2 = (loss_func_mask(img_output * weightmap2.unsqueeze(1),
                                       label[:, 0, :, :].unsqueeze(1) * weightmap2.unsqueeze(1))) / torch.sum(
                weightmap2)
            los_seg = loss_seg_dc(seg_output, seg_map)
            los = l1los_all + l1los_w1 + l1los_w2 + los_seg
            test_loss+=float(los.data)
            test_loss_l1+=float(l1los_all.data)
            test_loss_seg+=float(los_seg.data)
            test_loss_l1_w1+=float(l1los_w1.data)
            best_loss = test_loss

        print('Epoch:', epoch, '|Step:', i,
          '|train loss:', train_loss/step, '|test loss:', test_loss/i,'|l1 loss:', test_loss_l1/i,
              '|l1 loss on roi:', test_loss_l1_w1/i,'|segmentation loss:', test_loss_seg/i)

        if epoch%5 == 0 and best_loss <= test_loss:
            torch.save(net.state_dict(), './ckpt/best_model.pth')

    return train_loss/step,train_loss_l1/step,train_loss_l1_w1/step,train_loss_seg/step,\
           test_loss/i,test_loss_l1/i,test_loss_l1_w1/i,test_loss_seg/i



def TrainUnet(train_loader, test_loader,
                                         EPOCH = 200,batch_size=16,lr=0.001):

    vis = Visdom() #use_incoming_socket=False
    assert vis.check_connection()
    win_loss = vis.line(np.arange(10))  # create the window
    win_lossl1 = vis.line(np.arange(10))  # create the window
    win_lossl1curve = vis.line(np.arange(10))  # create the window
    win_lossseg = vis.line(np.arange(10))  # create the window
    x_index = []
    loss = [[], [], [], [], [], [], [], []]


    net = Models.R2U_Netseg(2)

    net = torch.nn.parallel.DataParallel(net)
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(EPOCH):
        train_loss, train_loss_l1, train_loss_l1_curve, train_loss_seg, test_loss, test_loss_l1, test_loss_l1_curve, test_loss_seg \
            = trainEpoch(epoch, train_loader, test_loader, net, optimizer, loss_func)
        x_index.append(epoch)
        loss[0].append(train_loss)
        loss[1].append(test_loss)

        loss[2].append(train_loss_l1)
        loss[3].append(test_loss_l1)

        loss[4].append(train_loss_l1_curve)
        loss[5].append(test_loss_l1_curve)

        loss[6].append(train_loss_seg)
        loss[7].append(test_loss_seg)

        vis.line(X=np.column_stack((np.array(x_index) for i in range(2))),
                 Y=np.column_stack((np.array(loss[i]) for i in range(2))),
                 win=win_loss,
                 opts=dict(title='LOSS',
                           xlabel='epoch',
                           xtick=1,
                           ylabel='loss',
                           markersymbol='dot',
                           markersize=5,
                           legend=['train loss', 'test loss']))
        vis.line(X=np.column_stack((np.array(x_index) for i in range(2))),
                 Y=np.column_stack((np.array(loss[i]) for i in [2, 3])),
                 win=win_lossl1,
                 opts=dict(title='LOSS',
                           xlabel='epoch',
                           xtick=1,
                           ylabel='loss',
                           markersymbol='dot',
                           markersize=5,
                           legend=['train loss l1', 'test loss l1']))
        vis.line(X=np.column_stack((np.array(x_index) for i in range(2))),
                 Y=np.column_stack((np.array(loss[i]) for i in [4, 5])),
                 win=win_lossl1curve,
                 opts=dict(title='LOSS',
                           xlabel='epoch',
                           xtick=1,
                           ylabel='loss',
                           markersymbol='dot',
                           markersize=5,
                           legend=['train loss l1 curve', 'test loss l1 curve']))
        vis.line(X=np.column_stack((np.array(x_index) for i in range(2))),
                 Y=np.column_stack((np.array(loss[i]) for i in [6, 7])),
                 win=win_lossseg,
                 opts=dict(title='LOSS',
                           xlabel='epoch',
                           xtick=1,
                           ylabel='loss',
                           markersymbol='dot',
                           markersize=5,
                           legend=['train loss seg', 'test loss sef']))


if __name__ == "__main__":
    train_data = biDataset(filename='datalist.csv',idxLeft=0,idxRight=7727)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_data = biDataset(filename='datalist.csv',idxLeft=7728,idxRight=12420)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    TrainUnet(train_loader,test_loader)