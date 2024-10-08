from random import random

import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
# from time import time
import time
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)# 这里出问题
    pre, recall, ndcg, hit = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        hit.append(ret['hit'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'hit': np.array(hit),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict  # 测试的地方
    testDict1: dict = dataset.testDict1  # 女测试的地方

    Recmodel: model.LightGCN
    adj_mat = dataset.UserItemNet.tolil()
    adj_mat1 = dataset.UserItemNet1.tolil()

    if (world.simple_model == 'lgn-ide'):
        lm = model.LGCN_IDE(adj_mat)
        lm.train()
    elif (world.simple_model == 'gf-cf'):
        lm = model.GF_CF(adj_mat)
        lm.train()
    elif world.simple_model == 'hbra':
        meta_path_net = dataset.dict_meta_path_net
        lm = model.HBRA(meta_path_net)
        lm.train()
        meta_path_net1 = dataset.dict_meta_path_net1
        lm1 = model.HBRA(meta_path_net1)
        lm1.train()
    elif (world.simple_model == 'bspm'):
        lm = model.BSPM(adj_mat, world.config)
        lm.train()
    elif (world.simple_model == 'bspm-torch'):
        lm = model.BSPM_TORCH(adj_mat, world.config)
        lm.train()
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'hit': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())  # 男
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        start = time.time()
        total_time = 0

        if world.simple_model == 'hbra':
            batch_test = {}
            batch_test['user'] = convert_sp_mat_to_sp_tensor(adj_mat).to('cuda:0')
            batch_test['item'] = convert_sp_mat_to_sp_tensor(adj_mat.T).to('cuda:0')
            all_rating = lm.getUsersRating(batch_test, world.dataset) #出问题处
            print(f'{all_rating}----------------------------------------') #将getusersR改下变量user改成item就好了

        #f = open('ans1.txt', 'w+')
        """
        for i in range(all_rating.shape[0]):
            for j in range(all_rating.shape[1]):
                if all_rating[i][j]>=1.0:
                    print(f"(i,j):{i},{j} {all_rating[i][j]}")
                    f.write(str(i)+' '+str(j)+ ' '+ str(all_rating[i][j])+'\n')
        
        print("*******************************************")
        """
        i = 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            print(f'{i} / {total_batch}')
            i = i + 1
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)# 用于计时
            starter.record()# 记录开始时间
            allPos = dataset.getUserPosItems(batch_users)#获取他们对应的标签，保存到,获取的是对的 训练
            print(f"allpos{allPos}")
            groundTrue = [testDict[u] for u in batch_users] # 测试
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            if world.simple_model == 'hbra':
                #print(f"{batch_users}-----")
                rating = all_rating[batch_users]
                rating = rating.to(world.device)
            elif (world.simple_model in ['gf-cf', 'bspm']):
                rating = lm.getUsersRating(batch_users, world.dataset)
                rating = torch.from_numpy(rating)
                rating = rating.to(world.device)
            elif (world.simple_model == 'bspm-torch'):
                if not torch.is_tensor(adj_mat):
                    adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()
                batch_ratings = adj_mat[batch_users, :].to(world.device)
                rating = lm.getUsersRating(batch_ratings, world.dataset)
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
                #print(f"{range_i},{exclude_items}")
                #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            rating[exclude_index, exclude_items] = -(1 << 10)# 出问题！！！！有测试集了
            _, rating_K = torch.topk(rating, k=max_K)
            #print(f"{rating_K}+++++++++")
            #for s in range(len(batch_users)):
            #    f.write(str(batch_users[s])+" ")
            #    p = int(rating_K[s][0])
            #    f.write(str(p))
            #    f.write("\n")

            rating = rating.cpu().numpy()

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            #print(f'{rating_list}-------------------')
        assert total_batch == len(users_list)
        end = time.time()
        print_time = False
        if print_time == True:
            print('inference time: ', end - start)
            print('inference time(CUDA): ', total_time)
        X = zip(rating_list, groundTrue_list) #######这里出问题

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x)) ### 在这里出错了
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hit'] += result['hit']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hit'] /= float(len(users))
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)

        ######################################################## 女
        users = list(testDict1.keys())  # 女
        print(users)
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        start = time.time()
        total_time = 0
        i = 1
        if world.simple_model == 'hbra':
            batch_test = {}
            batch_test['user1'] = convert_sp_mat_to_sp_tensor(adj_mat1).to('cuda:0')
            batch_test['item1'] = convert_sp_mat_to_sp_tensor(adj_mat1.T).to('cuda:0')
            all_rating = lm1.getUsersRating(batch_test, world.dataset) # !!!!!!!!!!!!!!!!!!!!1
            #print(f'{all_rating.shape}----------------------------------------') #将getusersR改下变量user改成item就好了

        # f = open('ans_woman.txt', 'w+')
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            print(f'{i} / {total_batch}')
            i = i + 1
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)# 用于计时
            starter.record()# 记录开始时间
            allPos = dataset.getUserPosItems1(batch_users)#获取他们对应的标签，保存到,获取的是对的 训练 这个!!!!!!!!!!!!!!!!
            #print("debug")
            #print(f"allpos{allPos}")
            groundTrue = [testDict1[u] for u in batch_users] # 测试
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            if world.simple_model == 'hbra':
                #print(f"{batch_users}-----")
                rating = all_rating[batch_users]
                rating = rating.to(world.device)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)# 出问题！！！！有测试集了
            _, rating_K = torch.topk(rating, k=max_K)
            print(f"{rating_K}+++++++++")
            #for s in range(len(batch_users)):
            #    f.write(str(batch_users[s])+" ")
            #    ss=random()
            #     if ss>=0.25:
            #         p = int(rating_K[s][0])
            #         f.write(str(p))
            #     else:
            #         aa=random()
            #         p = int(320*aa)
            #         f.write(str(p))
            #     f.write("\n")

            rating = rating.cpu().numpy()

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            #print(f'{rating_list}-------------------')
        assert total_batch == len(users_list)
        end = time.time()
        print_time = False
        if print_time == True:
            print('inference time: ', end - start)
            print('inference time(CUDA): ', total_time)
        X = zip(rating_list, groundTrue_list) #######这里出问题

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x)) ### 在这里出错了
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hit'] += result['hit']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hit'] /= float(len(users))
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        #f.close()


        return results

def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
