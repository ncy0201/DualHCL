import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def greedy_match(S):
    """
    :param S: Scores matrix, shape MxN where M is the number of source nodes,
        N is the number of target nodes.
    :return: A dict, map from source to list of targets.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m,n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:, :k]
    #top = np.argsort()[:, ::-1]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx,elm] = 1
    return result


def get_statistics(alignment_matrix, groundtruth_matrix):
    results = {}
    pred = greedy_match(alignment_matrix)
    greedy_match_acc = compute_accuracy(pred, groundtruth_matrix)
    results['Acc'] = float("{:.4f}".format(greedy_match_acc))
    # MAP = compute_MAP(alignment_matrix, groundtruth_matrix)
    # print("MAP: %.4f" % MAP)

    MRR, AUC, Hit = compute_MRR_AUC_Hit(alignment_matrix, groundtruth_matrix)
    results['MRR'] = float("{:.4f}".format(MRR))
    results['AUC'] = float("{:.4f}".format(AUC))
    results['Hit'] = float("{:.4f}".format(Hit))

    pred_top_1 = top_k(alignment_matrix, 1)
    precision_1 = compute_precision_k(pred_top_1, groundtruth_matrix)

    pred_top_5 = top_k(alignment_matrix, 5)
    precision_5 = compute_precision_k(pred_top_5, groundtruth_matrix)

    pred_top_10 = top_k(alignment_matrix, 10)
    precision_10 = compute_precision_k(pred_top_10, groundtruth_matrix)

    pred_top_15 = top_k(alignment_matrix, 15)
    precision_15 = compute_precision_k(pred_top_15, groundtruth_matrix)

    pred_top_20 = top_k(alignment_matrix, 20)
    precision_20 = compute_precision_k(pred_top_20, groundtruth_matrix)

    pred_top_25 = top_k(alignment_matrix, 25)
    precision_25 = compute_precision_k(pred_top_25, groundtruth_matrix)
    
    pred_top_30 = top_k(alignment_matrix, 30)
    precision_30 = compute_precision_k(pred_top_30, groundtruth_matrix)

    results['Precision@1'] = float("{:.4f}".format(precision_1))
    results['Precision@5'] = float("{:.4f}".format(precision_5))
    results['Precision@10'] = float("{:.4f}".format(precision_10))
    results['Precision@15'] = float("{:.4f}".format(precision_15))
    results['Precision@20'] = float("{:.4f}".format(precision_20))
    results['Precision@25'] = float("{:.4f}".format(precision_25))
    results['Precision@30'] = float("{:.4f}".format(precision_30))
    return results

def compute_precision_k(top_k_matrix, gt):
    n_matched = 0
    gt_candidates = np.argmax(gt, axis = 1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_accuracy(greedy_matched, gt):
    # print(gt)
    n_matched = 0
    for i in range(greedy_matched.shape[0]):
        if greedy_matched[i].sum() > 0 and np.array_equal(greedy_matched[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    print("已匹配点数：" + str(n_matched))
    print("应匹配点数：" + str(n_nodes))
    return n_matched/n_nodes


def compute_MRR_AUC_Hit(alignment_matrix, gt):
    S_argsort = alignment_matrix.argsort(axis=1)[:, ::-1]
    m = gt.shape[1] - 1
    MRR = 0
    AUC = 0
    Hit = 0
    for i in range(len(S_argsort)):
        predicted_source_to_target = S_argsort[i]
        # true_source_to_target = gt[i]
        for j in range(gt.shape[1]):
            if gt[i, j] == 1:
                for k in range(len(predicted_source_to_target)):
                    if predicted_source_to_target[k] == j:
                        ra = k + 1
                        MRR += 1/ra
                        AUC += (m+1-ra)/m
                        Hit += (m+2-ra)/(m+1)
                        break
                break
    n_nodes = (gt==1).sum()
    MRR /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MRR, AUC, Hit

# Input: positive test edges, negative test edges, edge score matrix
# Output: ROC AUC score,  AP score
def get_roc_score(edges_pos, edges_neg, score_matrix):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]])

    # Store negative edge predictions, actual values
    preds_neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]])

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])  # 按水平方向拼接数组
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # return roc_score, ap_score
    return roc_score, ap_score