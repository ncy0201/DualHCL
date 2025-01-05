import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import os
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_edgeindex(edgepath:str)->torch.tensor:
    '''return pyG version of edge_index'''
    edgelist=np.loadtxt(edgepath)
    edge = np.unique(np.sort(edgelist, axis=1), axis=0).T
    edge_index=torch.LongTensor(edge)
    edge_index_u=torch.vstack((edge_index[1],edge_index[0]))
    edge_index=torch.hstack((edge_index,edge_index_u))
    return edge_index

def get_gt_matrix(path,shape):
    '''return ground truth matrix'''
    gt = np.zeros(shape)
    with open(path) as file:
        for line in file:
            s, t = line.strip().split()
            gt[int(s),int(t)] = 1
    return gt

def get_dual_hyper(g: nx.Graph, x: torch.tensor)->Data:
    '''return dual hypergraph'''
    hyper_edge = torch.tensor(list(g.edges())).view(1,-1)
    hyper_node = torch.arange(0, g.number_of_edges(),1).repeat_interleave(2).view(1,-1)
    hyper_index = torch.cat([hyper_node,hyper_edge],dim=0).long()
    node_edge_map = torch.tensor(list(g.edges()), dtype=torch.long)
    # x = torch.FloatTensor(x)
    # x = F.normalize(x, p=2, dim=1)
    hyper_x = x[node_edge_map[:, 0]] * x[node_edge_map[:, 1]]
    hyper_x = F.normalize(hyper_x, p=2, dim=1)
    pyg_hyper = Data(x=hyper_x, edge_index=hyper_index, label=node_edge_map)
    return pyg_hyper

# def get_linegraph(g: nx.Graph)->nx.Graph:
#     '''return line graph'''
#     lg = nx.line_graph(g)
#     for i, node in enumerate(lg.nodes()):
#         # 保存节点的映射关系
#         lg.nodes[node]['label'] = torch.tensor(node, dtype=torch.long)
#     return lg

# def node_to_edge(l: nx.Graph, x: np.ndarray)->np.ndarray:
#     '''get edge feature from node feature'''
#     edge = np.array(list(l.nodes()))
#     s, t = edge[:, 0], edge[:, 1]
#     x = x[s] * x[t]
#     return x


# def nx_to_pyg(g: nx.graph, x: np.ndarray)->Data:
#     '''return pyG version of graph'''
#     x = torch.FloatTensor(x)
#     # x = F.normalize(x, p=2, dim=1) # L2 normalization
#     for i, node in enumerate(g.nodes()):    
#         g.nodes[node]['x'] = x[i]
#     pyg = from_networkx(g)
#     if hasattr(pyg, 'weight'):
#         del pyg.weight
#     return pyg

def nx_to_pyg(g: nx.graph, x: np.ndarray)->Data:
    '''return pyG version of graph'''
    x = torch.FloatTensor(x)
    edge = np.array(list(g.edges())).T
    edge_index=torch.LongTensor(edge)
    edge_index_u=torch.vstack((edge_index[1],edge_index[0]))
    edge_index=torch.hstack((edge_index,edge_index_u))
    # for i, node in enumerate(g.nodes()):    
    #     g.nodes[node]['x'] = x[i]
    # pyg = from_networkx(g)
    # if hasattr(pyg, 'weight'):
    #     del pyg.weight
    pyg = Data(x=x, edge_index=edge_index)
    return pyg

def get_adj(edgepath:str)->np.array:
    '''return adjacency matrix'''
    g = nx.read_edgelist(edgepath,nodetype=int)
    assert max(g.nodes())+1 == g.number_of_nodes(), "Graph contains isoload nodes"
    # adjacency = np.zeros((len(g.nodes()), len(g.nodes())))
    # for src_id, trg_id in g.edges():
    #     adjacency[src_id, trg_id] = 1
    #     adjacency[trg_id, src_id] = 1
    adjacency = nx.to_numpy_array(g, nodelist=sorted(g.nodes()))
    return adjacency

def load_embeddings(filename:str)->np.array:
    '''load node embedding'''
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    embedding=np.zeros((node_num, size))
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        embedding[int(vec[0])] = [float(x) for x in vec[1:]]
    fin.close()
    return embedding

def to_word2vec_format(val_embeddings, out_dir, filename, pref=""):
    '''
    save embeddings to 'out_dir/filename.txt'
    '''
    val_embeddings = val_embeddings.detach().cpu().numpy()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num=val_embeddings.shape[0]
    dim=val_embeddings.shape[1]

    with open("{0}/{1}".format(out_dir, filename), 'w') as f_out:
        f_out.write("%s %s\n"%(num, dim))
        for node in range(num):
            txt_vector = ["%s" % val_embeddings[node][j] for j in range(dim)]
            f_out.write("%s%s %s\n" % (pref, node, " ".join(txt_vector)))
        f_out.close()
    print("Emb has been saved to: {0}/{1}".format(out_dir, filename))

def init_log(name, logging_path='all.log'):
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 文件输出
    # file_handler = logging.FileHandler(filename=logging_path + "/all.log", encoding='utf-8')
    file_handler = RotatingFileHandler(filename=logging_path, maxBytes=1024 * 1024 * 10, backupCount=5, encoding='utf-8')
    # file_handler = TimedRotatingFileHandler(filename=logging_path + "/all.log", when='D', backupCount=7, encoding='utf-8')
    logger.addHandler(file_handler)

    # 控制台输出
    # console_handler = logging.StreamHandler()
    # logger.addHandler(console_handler)

    # 格式化
    # fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] : %(message)s'
    formatter = logging.Formatter(fmt, datefmt='%m-%d %H:%M:%S')
    
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)
    
    return logger

