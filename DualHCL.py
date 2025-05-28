import torch
import numpy as np
import time
import torch.nn.functional as F
from matcher import get_statistics, top_k, compute_precision_k
from utils import get_adj, get_gt_matrix, load_embeddings, get_dual_hyper, get_edgeindex
from node2vec import node2vec
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, HypergraphConv
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import networkx as nx
from tqdm import tqdm




def parse_args():
    parser = argparse.ArgumentParser(description="DualHCL")
    parser.add_argument('--gt_path', default='./data/douban/node,split=0.2.test.dict')
    parser.add_argument('--train_path', default='./data/douban/node,split=0.2.train.dict')
    parser.add_argument('--out_path', default='./data/douban/embeddings')
    parser.add_argument('--s_edge', default='./data/douban/online.txt')
    parser.add_argument('--t_edge', default='./data/douban/offline.txt')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--alpha', default=0.2, type=float)
    parser.add_argument('--tau', default=0.2, type=float)
    parser.add_argument('--neg', default=5, type=int)
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return parser.parse_args()

class DualHCL(torch.nn.Module):
    def __init__(self, args, s_input=128, t_input=128):
        super().__init__()
        s_input = s_input
        t_input = t_input
        output = args.dim
        # orignal graph
        self.conv1 = GCNConv(s_input, 2 * output)
        self.conv2 = GCNConv(t_input, 2 * output)
        self.conv3 = GCNConv(2 * output, output)
        self.activation = nn.ReLU()
        # dual hypergraph
        self.dual1 = HypergraphConv(s_input, 2 * output)
        self.dual2 = HypergraphConv(t_input, 2 * output)
        self.dual3 = HypergraphConv(2 * output, output)
        self.dual_activation = nn.ReLU()
        
        self.args = args

    def s_forward(self, s):
        '''G^s'''
        x = self.conv1(s.x, s.edge_index)
        x = self.activation(x)
        return self.conv3(x, s.edge_index)
    
    def s_dual(self, s_dual):
        ''''H^s'''
        x = self.dual1(s_dual.x, s_dual.edge_index)
        x = self.dual_activation(x)
        return self.dual3(x, s_dual.edge_index)

    def t_forward(self, t):
        '''G^t'''
        x = self.conv2(t.x, t.edge_index)
        x = self.activation(x)
        return self.conv3(x, t.edge_index)
    
    def t_dual(self, t_dual):
        '''H^t'''
        x = self.dual2(t_dual.x, t_dual.edge_index)
        x = self.dual_activation(x)
        return self.dual3(x, t_dual.edge_index)


    def single_recon_loss(self, z, pos_edge_index, neg_edge_index):
        '''single layer reconstruction loss'''
        # avoid zero when calculating logarithm
        EPS = 1e-15  
        # loss of positive samples
        pos_dot = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))
        pos_loss = -torch.log(pos_dot + EPS).mean()  
        # loss of negative samples
        neg_dot = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))
        neg_loss = -torch.log(1 - neg_dot + EPS).mean()  
        return pos_loss + neg_loss

    def recon_loss(self, zs, zt, s_pos_edge_index, t_pos_edge_index):
        '''reconstruction loss to preserve intra-network structural features'''
        # negative sampling
        neg_s = negative_sampling(s_pos_edge_index, zs.shape[0])
        neg_t = negative_sampling(t_pos_edge_index, zt.shape[0])
        # 
        s_recon_loss = self.single_recon_loss(zs, s_pos_edge_index, neg_s)
        t_recon_loss = self.single_recon_loss(zt, t_pos_edge_index, neg_t)
        return s_recon_loss + t_recon_loss
    
    def inter_loss(self, zs, zt, gt):
        '''inter-network loss to preserve anchor node structural features'''
        s_a = zs[gt[:, 0]]
        t_a = zt[gt[:, 1]]
        
        loss = self.info_NCE(s_a, t_a, self.args.tau) + self.info_NCE(t_a, s_a, self.args.tau)
        return loss / 2.0

    
    def node_edge_CL(self, s_x, s_h, t_x, t_h, s_ce, t_ce, s_dual, t_dual):
        '''edge contrastive Loss'''
        # get confidence edge from dual hypergraph corresponding node
        s_edgeindex = s_dual.label[s_ce]
        t_edgeindex = t_dual.label[t_ce]
        # get edge feature from node feature: Hadamard product
        s_x_l = s_x[s_edgeindex[:, 0]] * s_x[s_edgeindex[:, 1]]
        t_x_l = t_x[t_edgeindex[:, 0]] * t_x[t_edgeindex[:, 1]]
        
        s_x_l = F.normalize(s_x_l, p=2, dim=1)
        t_x_l = F.normalize(t_x_l, p=2, dim=1)
        # contrastive loss
        ne_loss_s = self.info_NCE(s_h[s_ce], s_x_l, tau=self.args.tau) + self.info_NCE(s_x_l, s_h[s_ce], tau=self.args.tau)
        
        ne_loss_t = self.info_NCE(t_h[t_ce], t_x_l, tau=self.args.tau) + self.info_NCE(t_x_l, t_h[t_ce], tau=self.args.tau)
        return (ne_loss_s+ne_loss_t)/2.0
    
    def edge_node_CL(self, s_x, s_x_n, t_x, t_x_n):
        '''node contrastive Loss'''
        en_loss_s = self.info_NCE(s_x_n, s_x, tau=self.args.tau) + self.info_NCE(s_x, s_x_n, tau=self.args.tau)
        
        en_loss_t = self.info_NCE(t_x_n, t_x, tau=self.args.tau) + self.info_NCE(t_x, t_x_n, tau=self.args.tau)
        return (en_loss_s+en_loss_t)/2.0
    
    def info_NCE(self, emb_s, emb_t, tau=1.0):
        '''calculate infoNCE loss'''
        N, _ = emb_s.shape
        neg_num = self.args.neg
    
        # all pairwise similarity
        emb_s_norm = F.normalize(emb_s, p=2, dim=1)
        emb_t_norm = F.normalize(emb_t, p=2, dim=1)
        sim_matrix = torch.matmul(emb_s_norm, emb_t_norm.T) / tau
        
        # positive samples
        pos_sims = torch.diag(sim_matrix).unsqueeze(1)
        # negative samples
        neg_indices = torch.randint(0, N, (N, neg_num), device=emb_s.device)
        neg_sims = torch.gather(sim_matrix, 1, neg_indices)
        
        # concatenate positive and negative samples
        logits = torch.cat([pos_sims, neg_sims], dim=1)
        
        # positive samples labels
        labels = torch.zeros(N, dtype=torch.long, device=emb_s.device)
        
        # infoNCE loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


def get_embedding(s_pyg, t_pyg, s_dual, t_dual, \
    g_s, g_t, train_anchor, groundtruth_matrix, args):
    model = DualHCL(args).to(args.device)
    # get high confidence edges index
    s_ce, t_ce = confidence_edges(g_s, g_t, s_dual, t_dual)
    # incidence matrix
    s_inc = torch.from_numpy(nx.incidence_matrix(g_s).todense()).float().to(args.device)
    t_inc = torch.from_numpy(nx.incidence_matrix(g_t).todense()).float().to(args.device)
    s_inc = F.normalize(s_inc, p=1, dim=1)
    t_inc = F.normalize(t_inc, p=1, dim=1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 

    # pbar = tqdm(total=args.epochs, desc="Training", file=sys.stdout)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        # orignal graph 
        zs = model.s_forward(s_pyg)
        zt = model.t_forward(t_pyg)
        # dual hypergraph 
        ds = model.s_dual(s_dual)
        dt = model.t_dual(t_dual)
        # get new node embedding from dual hypergraph
        s_x_n = torch.matmul(s_inc, ds)
        t_x_n = torch.matmul(t_inc, dt)
        # reconstruct loss
        intra_loss = model.recon_loss(zs, zt, s_pyg.edge_index, t_pyg.edge_index)
        # anchor node contrastive loss
        inter_loss1 = model.inter_loss(zs, zt, train_anchor)
        inter_loss2 = model.inter_loss(s_x_n, t_x_n, train_anchor)
        # node-edge contrastive loss
        node_edge_loss = model.node_edge_CL(zs, ds, zt, dt, s_ce, t_ce, s_dual, t_dual)
        edge_node_loss = model.edge_node_CL(zs, s_x_n, zt, t_x_n)
        ne_cl = node_edge_loss + edge_node_loss
        # total loss
        loss = (1 - args.alpha) * (intra_loss + inter_loss1) + args.alpha * (ne_cl + inter_loss2)
        loss.backward()
        optimizer.step()
        
        # pbar.update(1)
        # if (epoch+1) % 10 == 0:
        #     p1, p10 = evaluate(zs, zt, groundtruth_matrix)
        #     print(f'p1 : {p1}, p10 : {p10}')
    
    model.eval()
    s_embedding = model.s_forward(s_pyg)
    t_embedding = model.t_forward(t_pyg)
    s_embedding = s_embedding.detach().cpu()
    t_embedding = t_embedding.detach().cpu()
    return s_embedding, t_embedding

@torch.no_grad()
def evaluate(zs, zt, gt):
    '''
    calculate Precision@k for evaluation
    '''
    z1 = zs.detach().cpu()
    z2 = zt.detach().cpu()
    S = cosine_similarity(z1, z2)
    pred_top_1 = top_k(S, 1)
    precision_1 = compute_precision_k(pred_top_1, gt)
    
    pred_top_10 = top_k(S, 10)
    precision_10 = compute_precision_k(pred_top_10, gt)
    return precision_1, precision_10

def confidence_edges(gs, gt, s_dual, t_dual):
    '''sample high confidence edges'''
    s_edgeindex = s_dual.label
    t_edgeindex = t_dual.label
    # G_s
    s_degree = torch.tensor([d for _, d in gs.degree()]).to(s_edgeindex.device)
    s_edge_scores = s_degree[s_edgeindex[:, 0]] * s_degree[s_edgeindex[:, 1]]
    ps = gs.number_of_nodes() / gs.number_of_edges()
    s_num_edges_to_keep = int(len(s_edgeindex) * ps)
    _, s_indices = torch.sort(s_edge_scores, descending=True)
    s_ce = s_indices[:s_num_edges_to_keep]
    # G_t
    t_degree = torch.tensor([d for _, d in gt.degree()]).to(t_edgeindex.device)
    t_edge_scores = t_degree[t_edgeindex[:, 0]] * t_degree[t_edgeindex[:, 1]]
    pt = gt.number_of_nodes() / gt.number_of_edges()
    t_num_edges_to_keep = int(len(t_edgeindex) * pt)
    _, t_indices = torch.sort(t_edge_scores, descending=True)
    t_ce = t_indices[:t_num_edges_to_keep]
    
    return s_ce, t_ce


if __name__ == "__main__":
    metrics = ['Acc', 'MRR', 'AUC', 'Hit', 'Precision@1', 'Precision@5', 'Precision@10', 'Precision@15', 'Precision@20', \
        'Precision@25', 'Precision@30', 'time']
    print(metrics)
    results = dict.fromkeys(('Acc', 'MRR', 'AUC', 'Hit', 'Precision@1', 'Precision@5', 'Precision@10', 'Precision@15', \
        'Precision@20', 'Precision@25', 'Precision@30', 'time'), 0.0) # save results
    args = parse_args()
    # two graphs
    s_adj = get_adj(args.s_edge)
    t_adj = get_adj(args.t_edge)
    s_e = get_edgeindex(args.s_edge)
    t_e = get_edgeindex(args.t_edge)
    g_s = nx.from_numpy_array(s_adj)
    g_t = nx.from_numpy_array(t_adj)
    s_num = s_adj.shape[0]
    t_num = t_adj.shape[0]
    # train set
    train_anchor = torch.LongTensor(np.loadtxt(args.train_path, dtype=int))
    # test set
    groundtruth_matrix = get_gt_matrix(args.gt_path, (s_num, t_num))

    # repeat times for average, default: 1
    for i in range(args.N):
        print('Iteration: %d' % (i+1))
        print('Generate deepwalk embeddings as input X...')
        s_x = node2vec(s_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        t_x = node2vec(t_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        # s_x = load_embeddings(args.out_path + f'/source_emb_{i}')
        # t_x = load_embeddings(args.out_path + f'/target_emb_{i}')
        s_x = torch.FloatTensor(s_x)
        t_x = torch.FloatTensor(t_x)
        # s and t pyg.Data
        s_pyg = Data(x=s_x, edge_index=s_e).to(args.device)
        t_pyg = Data(x=t_x, edge_index=t_e).to(args.device)
        # s_dual and t_dual pyg.Data
        s_dual = get_dual_hyper(g_s, s_x).to(args.device)
        t_dual = get_dual_hyper(g_t, t_x).to(args.device)
        
        start_time = time.time()
        
        print('Generate embeddings...')
        s_embedding, t_embedding= get_embedding(s_pyg, t_pyg, s_dual, t_dual, g_s, g_t, train_anchor, groundtruth_matrix, args)

        print('Evaluating...')
        S = cosine_similarity(s_embedding, t_embedding)
        result = get_statistics(S, groundtruth_matrix)
        t = time.time() - start_time
        for k, v in result.items():
            results[k] += v
        vector = [result[metric] for metric in metrics if metric in result]
        print(', '.join(f'{v:.4f}' for v in vector))

        results['time'] += t
    for k in results.keys():
        results[k] /= args.N
        
    print('\nDualHCL')
    print(args)
    print('Average results:')
    vector = [results[metric] for metric in metrics if metric in results]
    print(', '.join(f'{v:.4f}' for v in vector))