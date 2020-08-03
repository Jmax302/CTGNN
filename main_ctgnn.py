import argparse
import os
import time
from warnings import simplefilter

import scipy.sparse as sp
import tensorflow as tf
from tqdm import tqdm

from model_ctgnn import Model
from sampler_time_gcn import WarpSampler
from util import *

simplefilter(action='ignore', category=FutureWarning)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ctgnn_clothes')
parser.add_argument('--train_dir', default='data/ctgnn/')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=15, type=int)
parser.add_argument('--hidden_units', default=128, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--gpu', default='0')
parser.add_argument('--test_epoch', default=5, type=int)
parser.add_argument('--candidate_count', default=100, type=int)
parser.add_argument('--info', default='none')
parser.add_argument('--n_layers', default=5, type=int)

if __name__ == '__main__':
    simplefilter(action='ignore', category=FutureWarning)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('-----------------------------args----------------------------------')
    print('dataset:      ', args.dataset)
    print('maxlen:       ', args.maxlen)
    print('hiddenunits:  ', args.hidden_units)
    print('dropout_rate: ', args.dropout_rate)
    print('gpu_number:   ', args.gpu)
    print('blocks:       ', args.num_blocks)
    print('num_heads:    ', args.num_heads)
    print('information:  ', args.info)
    print('gcn_layers:   ', args.n_layers)
    print('lr:           ', args.lr)
    print('l2_emb:       ', args.l2_emb)
    print('--------------------------------------------------------------------')

    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(list(vars(args).items()), key=lambda x: x[0])]))
    f.close()

    # ---------------------load category idx--------------------------
    with open('data/ctgnn/' + args.dataset + '_category_idx.pk', 'rb')as f2:
        category_idx = pickle.load(f2)
        category_idx.insert(0, 0)
    category_idx_num = len(category_idx)
    # print(category_idx_num)

    norm_adj = sp.load_npz('data/ctgnn/adj_matrix/' + args.dataset + '/s_norm_adj_mat_time.npz')
    # norm_adj_time = sp.load_npz('data/ctgnn/adj_matrix/' + args.dataset + '/s_norm_adj_mat_time.npz')

    dataset = time_category_data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, categorynum, time_train, time_valid, time_test] = dataset
    # print(itemnum)
    # print(categorynum)

    num_batch = len(user_train) / args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=20,
                          time_train=time_train)
    model = Model(usernum, itemnum, args, norm_adj, category_idx_num, category_idx)
    sess.run(tf.initialize_all_variables())

    T = 0.0
    t0 = time.time()

    # u, seq, pos, neg, time_interval = sampler.next_batch()  #

    # try:
    for epoch in range(1, args.num_epochs + 1):
        for step in tqdm(list(range(int(num_batch))), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, time_interval, pos_mask = sampler.next_batch()  #
            # print(np.shape(time_interval))
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True, model.time_interval: time_interval,
                                     model.category_idx: category_idx})

        if epoch % args.test_epoch == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end=' ')
            t_test = evaluate_time_category(model, dataset, args, sess, category_idx)
            # t_valid = evaluate_valid(model, dataset, args, sess)
            t_valid = [0, 0]
            print('')
            print('epoch:%d, time: %f(s), test (NDCG@5: %.4f, HR@5: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                epoch, T, t_test[0], t_test[1], t_test[2], t_test[3]))

            f.write(str(t_test) + '\n')
            f.flush()
            t0 = time.time()


    f.close()
    sampler.close()
    print("Done")
