import networkx as nx

import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import torch
import itertools
import re
from sklearn.metrics import accuracy_score


def cace_score(c_pred_c0, c_pred_c1):
    cace = torch.abs(c_pred_c1.mean(dim=0) - c_pred_c0.mean(dim=0))
    return cace


def explanation_to_latex(explanation):
    latex_str = "\\begin{align}\n"

    # Sort items by keys and add to the LaTeX string
    for key in sorted(explanation.keys()):
        # Add the expression to the string, carefully removing only the standalone dollar signs
        expression = explanation[key].strip('$')
        latex_str += expression + " \\\\\n"

    # End the LaTeX align environment
    latex_str += "\\end{align}"
    return latex_str


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()
 


def orthogonality_loss(pre_emb):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    result = torch.zeros((pre_emb.shape[0], pre_emb.shape[1], pre_emb.shape[1]**2))
    for i in range(pre_emb.shape[1]):
        k = 0
        for j in range(0, pre_emb.shape[1]):
            result[:, i, k] = cos(pre_emb[:, i], pre_emb[:, j])
            k += 1
    return result.mean()

def evaluate_exp(explanations, s_preds):
    pattern = r'c_\d+'
    s_preds = (s_preds > 0.5).float()
    results = [0 for i in range(s_preds.shape[-1])]
    for i in range(s_preds.shape[0]):
        for j in range(s_preds.shape[-1]):
            pred = s_preds[i, j] 
            if str(explanations[j]) == 'True':
                exp_pred = 1
            elif explanations[j] is None:
                exp_pred = pred
            elif str(explanations[j]) == 'False':
                exp_pred = 0
            else:
                # Extract all terms that match the pattern
                extracted_terms = re.findall(pattern, str(explanations[j]))

                # Remove duplicates by converting the list to a set, then back to list if needed
                unique_terms = list(set(extracted_terms))
                symbols_dict = {symbol: bool(s_preds[i][int(symbol.split('_')[-1])]) for symbol in unique_terms}
                exp_pred = explanations[j].subs(symbols_dict)
                if exp_pred:
                    exp_pred = 1
                else:
                    exp_pred = 0
            results[j] += int(pred == exp_pred)
    results = np.array(results)/s_preds.shape[0]
    return results

##Entropy
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en


#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)

def conditional_entropy_dag(s):
    dag = np.zeros((s.shape[1], s.shape[1]))
    for i in range(s.shape[1]):
        for j in range(s.shape[1]):
            if i != j:
                dag[i, j] = cEntropy(s[:, i], s[:, j])
    return dag

def compute_ps_pn_matrix(x, model, dag):
    # matrix nxn zeros
    matrix_ps = np.zeros((model.n_symbols, model.n_symbols))
    matrix_ps[:] = np.nan
    matrix_ps = matrix_ps.tolist()
    matrix_pn = np.zeros((model.n_symbols, model.n_symbols))
    matrix_pn[:] = np.nan
    matrix_pn = matrix_pn.tolist()
    zero = torch.zeros((x.shape[0], model.n_symbols))
    one = torch.ones((x.shape[0], model.n_symbols))
    s_pred = (model.forward(x) > 0.5).float()
    py = s_pred.mean(dim=0)
    for i in range(model.n_symbols):
        s_pred_c1 = (model.forward(x, c=one, intervention_idxs=[i]) > 0.5).float()
        s_pred_c0 = (model.forward(x, c=zero, intervention_idxs=[i]) > 0.5).float()

        py_notx = s_pred_c0.mean(dim=0)
        py_x = s_pred_c1.mean(dim=0)
        pnoty_notx = 1 - py_notx
        

        row = dag[i]

        for j, el in enumerate(row):
            if el != 0:
                filterx = s_pred[:, i] == 1
                filtery = s_pred[:, j] == 1
                filter_xy = filterx * filtery
                pyx = s_pred[filter_xy].shape[0] / s_pred.shape[0]
                filter_notx = s_pred[:, i] == 0
                filter_noty = s_pred[:, j] == 0
                filter_notxnoty = filter_notx * filter_noty
                pnotynotx = s_pred[filter_notxnoty].shape[0] / s_pred.shape[0]
                # not_el_c = (1 - s_pred_c1[:, j]).mean()
                # not_el_not_c = (1 - s_pred_c0[:, j]).mean()
                # print(not_el_c, not_el_not_c)
                # pn = not_el_c / not_el_not_c
                # matrix_pn[i, j] = pn
                # el_c = s_pred_c1[:, j].mean()
                # el_not_c = s_pred_c0[:, j].mean()
                # print(el_c, el_not_c)
                # ps = el_c / el_not_c
                # matrix_ps[i, j] = ps
                pn_min = max(((py[j] - py_notx[j]) / pyx).item(), 0)
                pn_max = min(((pnoty_notx[j] - pnotynotx) / pyx).item(), 1)
                matrix_pn[i][j] = (np.around(pn_min, 2), np.around(pn_max, 2))
                ps_min = max(((py_x[j] - py[j]) / pnotynotx).item(), 0)
                ps_max = min(((py_x[j] - pyx) / pnotynotx).item(), 1)
                matrix_ps[i][j] = (np.around(ps_min, 2), np.around(ps_max, 2))
    return matrix_ps, matrix_pn

def compute_pns_matrix(x, model, dag):
    #create a graph from the adjacency matrix
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    # fill dag with ones where there is an indirected connection
    for i in range(dag.shape[0]):
        for j in range(dag.shape[1]):
            if i != j:
                if nx.has_path(G, i, j):
                        dag[i, j] = 1
    # matrix nxn zeros
    matrix_pns = np.zeros((model.n_symbols, model.n_symbols))
    matrix_pns[:] = np.nan
    matrix_pns = matrix_pns.tolist()
    zero = torch.zeros((x.shape[0], model.n_symbols))
    one = torch.ones((x.shape[0], model.n_symbols))
    # s_pred = (model.forward(x) > 0.5).float()
    # py = s_pred.mean(dim=0)
    for i in range(model.n_symbols):
        s_pred_c1 = (model.forward(x, c=one, intervention_idxs=[i]) > 0.5).float()
        s_pred_c0 = (model.forward(x, c=zero, intervention_idxs=[i]) > 0.5).float()

        py_notx = s_pred_c0.mean(dim=0)
        py_x = s_pred_c1.mean(dim=0)
        pnoty_notx = 1 - py_notx
        

        row = dag[i]

        for j, el in enumerate(row):
            if el != 0:
                pns_min = max((py_x[j] - py_notx[j]).item(), 0)
                pns_max = min(py_x[j].item(), pnoty_notx[j].item())
                matrix_pns[i][j] = (np.around(pns_min, 2), np.around(pns_max, 2))
    return matrix_pns

def interventions_from_root(dag, model, x, s, order=None, exclude=None, perturb=True):
    if exclude is None:
        exclude = [s.shape[1]-1]
    dag = (dag > 0.1).float().numpy()
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    # # fill dag with ones where there is an indirected connection
    for i in range(dag.shape[0]):
        for j in range(dag.shape[1]):
            if i != j:
                if nx.has_path(G, i, j):
                        dag[i, j] = 1
    connections = dag.sum(axis=1)
    print(connections)
    order_connections = np.flip(np.argsort(connections))
    parent_indices = {node: list(G.predecessors(node)) for node in list(nx.topological_sort(G))}
    
    # add random noise to x 
    x_perturbed = x.clone()
    x_perturbed = x_perturbed + torch.randn_like(x_perturbed) * 15
    for i in range(s.shape[1]):
        s_pred = model(x_perturbed)
        concept_accuracy = accuracy_score(s[:, i].ravel(), s_pred[:, i].ravel() > 0.5)
        print(i, concept_accuracy)
    s_pred = model(x_perturbed)
    concept_accuracy = accuracy_score(s.ravel(), s_pred.ravel() > 0.5)
    acc = []
    abs_acc = [concept_accuracy]
    int_idexes = []
    if order is None:
        print(order_connections)
        for current_node in order_connections:
            if current_node in exclude:
                continue
            else:
                int_idexes += [current_node]
                s_pred_int = model(x_perturbed, c=s, intervention_idxs=int_idexes)
                s_pred = model(x_perturbed)
                to_include = [i for i in range(s.shape[1]) if i not in int_idexes and i not in exclude]
                # for e in exclude:
                #     if e in to_include:
                #         to_include.remove(e)
                print(to_include)
                if to_include != []:
                    concept_accuracy = accuracy_score(s[:, to_include].ravel(), s_pred[:, to_include].ravel() > 0.5)
                    concept_accuracy_int = accuracy_score(s[:, to_include].ravel(), s_pred_int[:, to_include].ravel() > 0.5)
                    acc += [concept_accuracy_int- concept_accuracy]
                    concept_accuracy_abs = accuracy_score(s.ravel(), s_pred_int.ravel() > 0.5)
                    abs_acc += [concept_accuracy_abs]
    else:
        print(order)
        for current_node in order:
            if current_node in exclude:
                continue
            else:
                int_idexes += [current_node]
                print(int_idexes)
                s_pred_int = model(x_perturbed, c=s, intervention_idxs=int_idexes)
                s_pred = model(x_perturbed)
                to_include = [i for i in range(s.shape[1]) if i not in int_idexes and i not in exclude]
                # for e in exclude:
                #     if e in to_include:
                #         to_include.remove(e)
                print(to_include)
                if to_include != []:
                    concept_accuracy = accuracy_score(s[:, to_include].ravel(), s_pred[:, to_include].ravel() > 0.5)
                    concept_accuracy_int = accuracy_score(s[:, to_include].ravel(), s_pred_int[:, to_include].ravel() > 0.5)
                    acc += [concept_accuracy_int-concept_accuracy]
                    concept_accuracy_abs = accuracy_score(s.ravel(), s_pred_int.ravel() > 0.5)
                    abs_acc += [concept_accuracy_abs]
    return acc, abs_acc





            