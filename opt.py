import argparse

parser = argparse.ArgumentParser()


#控制实验的参数： run控制执行几轮； epochs控制每次执行训练多少次
parser.add_argument('--runs', type=int, default=20, help='runs')#20
parser.add_argument('--epochs', type=int, default=400, help='training epoch')#400

#消融实验参数，请注意使用两个最好的seeds
parser.add_argument('--ZINB_only_use', type=bool, default=False, help='ZINB loss')# 是否仅使用改进的ZINB loss
parser.add_argument('--Hard_only_use', type=bool, default=False, help='Hard-sample learning loss')# 是否仅使用Hard sample learning loss

#参数敏感性实验参数
parser.add_argument('--gamma', type=float, default=4, help='Hard-sample focusing factor gamma')#控制硬样本关注度的参数，范围从1-5
parser.add_argument('--alpha', type=float, default=10, help='high confidence rate')#控制两个loss的贡献参数 从{0.01,0.1,1,10,100}

#可以分析也可以不分析：学习率
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

# dataset
parser.add_argument('--device', type=str, default="cpu", help='device.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
# parser.add_argument('--cluster_num', type=int, default=7, help='cluster number')

# pre-process
parser.add_argument('--n_input', type=int, default=500, help='input feature dimension')
parser.add_argument('--t', type=int, default=2, help="filtering time of Laplacian filters")

# network
parser.add_argument('--tao', type=float, default=0.9, help='high confidence rate')
parser.add_argument('--dims', type=int, default=[1500], help='hidden unit')
parser.add_argument('--activate', type=str, default='ident', help='activate function')


# training


parser.add_argument('--acc', type=float, default=0, help='acc')
parser.add_argument('--nmi', type=float, default=0, help='nmi')
parser.add_argument('--ari', type=float, default=0, help='ari')
parser.add_argument('--f1', type=float, default=0, help='f1')


args = parser.parse_args()
