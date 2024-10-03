import torch
import networkx as nx

from causal_cbm.dagma import CausalLayer
import random
import numpy as np 
from sympy.logic import SOPform
from sympy import symbols, true, false

EPS = 1e-6

class ConceptEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
            concept_family=None,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * emb_size, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 2 * emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * emb_size, 1)
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)
        self.concept_family = concept_family

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, train=False, return_intervened=False):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        c_pred_tmp_list = []
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            if self.concept_family is None:
                c_pred = torch.sigmoid(c_pred)
            # Time to check for interventions
            c_int = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )
            if return_intervened:
                c_pred_list.append(c_int)
            else:
                c_pred_list.append(c_pred)

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_int + context_neg * (1 - c_int)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)


class CausalConceptEmbeddingLayer(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            n_classes,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
            gamma=10.0,
            probabilistic=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.emb_size = emb_size
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_symbols = n_concepts + n_classes

        # self.prior_cem = ConceptEmbedding(in_features, self.n_symbols, emb_size,
        #                                   active_intervention_values, inactive_intervention_values,
        #                                   intervention_idxs, training_intervention_prob)

        self.eq_model = CausalLayer(self.n_concepts, self.n_classes,
                                    [self.emb_size, self.emb_size*2], bias=False,
                                    gamma=gamma)
        
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_context_generators_mean = torch.nn.ModuleList()
        self.concept_context_generators_var = torch.nn.ModuleList()
        self.concept_prob_predictor = torch.nn.ModuleList()
        self.concept_prob_predictor_post = torch.nn.ModuleList()
        self.column_dag_predictor = torch.nn.ModuleList()
        self.probabilistic = probabilistic
        self.prior = []
        for i in range(self.n_symbols):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * emb_size, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
            self.concept_context_generators_mean.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * emb_size, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
            self.concept_context_generators_var.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * emb_size, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
            self.concept_prob_predictor.append(torch.nn.Sequential(
                torch.nn.Linear(2 * emb_size, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, 1)
            ))
            self.concept_prob_predictor_post.append(torch.nn.Sequential(
                torch.nn.Linear(2 * emb_size, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, 1)
            ))
            self.column_dag_predictor.append(torch.nn.Sequential(
                torch.nn.Linear(n_concepts+n_classes, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, n_concepts+n_classes),
                torch.nn.Sigmoid(),
            ))
            self.prior.append(torch.distributions.normal.Normal(torch.nn.Parameter(torch.zeros(2 * self.emb_size)), torch.nn.Parameter(torch.ones(2 * self.emb_size))))
        self.reshaper = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )

        self.parent_indices = None
        self.scm = None
        self.training_intervention_prob = training_intervention_prob

    def expand_network(self, new_indexes, dag):
        self.n_concepts += len(new_indexes)
        self.n_symbols = self.n_concepts + self.n_classes
        self.eq_model.fc1 = torch.nn.Linear(self.n_symbols, self.n_symbols, bias=False)
        self.eq_model.fc1.weight = torch.nn.Parameter(dag, requires_grad=False)
        self.eq_model.n_concepts = self.n_concepts
        self.eq_model.n_symbols = self.n_symbols
        self.eq_model.I = torch.eye(self.n_symbols)
        self.eq_model.mask = torch.ones(self.n_symbols, self.n_symbols)
        self.eq_model.mask = self.eq_model.mask - self.eq_model.I
        self.eq_model.mask[self.n_concepts:] = torch.zeros(self.n_classes, self.n_symbols)

        self.eq_model.edges_to_check = []
        self.eq_model.edge_matrix = torch.nn.Parameter(torch.zeros(self.n_symbols, self.n_symbols))


        for i in range(len(new_indexes)):
            self.concept_context_generators.insert(new_indexes[i], torch.nn.Sequential(
                torch.nn.Linear(self.in_features, 2 * self.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * self.emb_size, 2 * self.emb_size),
                torch.nn.LeakyReLU(),
            ))
            self.concept_context_generators_mean.insert(new_indexes[i], torch.nn.Sequential(
                torch.nn.Linear(self.in_features, 2 * self.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * self.emb_size, 2 * self.emb_size),
                torch.nn.LeakyReLU(),
            ))
            self.concept_context_generators_var.insert(new_indexes[i], torch.nn.Sequential(
                torch.nn.Linear(self.in_features, 2 * self.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * self.emb_size, 2 * self.emb_size),
                torch.nn.LeakyReLU(),
            ))
            self.concept_prob_predictor.insert(new_indexes[i], torch.nn.Sequential(
                torch.nn.Linear(2 * self.emb_size, self.emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.emb_size, 1),
            ))
            self.prior.insert(new_indexes[i], torch.distributions.normal.Normal(torch.nn.Parameter(torch.zeros(self.emb_size)), torch.nn.Parameter(torch.ones(self.emb_size))))
            # self.prior.insert(new_indexes[i], torch.distributions.normal.Normal(torch.zeros(self.emb_size), torch.ones(self.emb_size)))
            
        freeze_set_context = set(list(range(self.n_symbols)))
        for i in new_indexes:
            if i in freeze_set_context:
                freeze_set_context.remove(i)
            for j in range(self.n_symbols):
                if dag[j, i] == 1:
                    if j in freeze_set_context:
                        freeze_set_context.remove(j)
        freeze_set_predictor = set(list(range(self.n_symbols)))
        for i in range(self.n_symbols):
            if i in new_indexes:
                if i in freeze_set_predictor:
                    freeze_set_predictor.remove(i)
            else:
                for j in new_indexes:
                    if dag[j, i] == 1:
                        if i in freeze_set_predictor:
                            freeze_set_predictor.remove(i)
                        for k in range(self.n_symbols):
                            if dag[k, i] == 1:
                                if k in freeze_set_context:
                                    freeze_set_context.remove(k)

        # freeze selected layers
        for i in range(self.n_symbols):
            if i in freeze_set_context:
                for param in self.concept_context_generators[i].parameters():
                    param.requires_grad = False
            if i in freeze_set_predictor:
                for param in self.concept_prob_predictor[i].parameters():
                    param.requires_grad = False

    def _build_concept_embedding(self, context_mix, c_pred):
        context_pos = context_mix[:, :self.emb_size]
        context_neg = context_mix[:, self.emb_size:]
        return context_pos * c_pred + context_neg * (1 - c_pred)

    def compute_parent_indices(self):
        self.scm = torch.FloatTensor(self.eq_model.fc1_to_adj())
        scm = self.scm.detach().numpy()
        check = False
        while check == False:
            try:
                G = nx.from_numpy_array(scm, create_using=nx.DiGraph)
                self.parent_indices = {node: list(G.predecessors(node)) for node in list(nx.topological_sort(G))}
                check = True
            except:
                scm_tmp = scm.copy()
                if self.eq_model.family_dict is not None:
                    rows = []
                    for key, value in self.eq_model.family_dict.items():
                        rows += [np.expand_dims(scm[value[0], :], axis=0)]
                    rows = np.concatenate(rows, axis=0)
                    cols = []
                    for key, value in self.eq_model.family_dict.items():
                        cols += [np.expand_dims(rows[:, value[0]], axis=1)]
                    scm_tmp = np.concatenate(cols, axis=1)
                # self.dag = scm_tmp.copy()
                scm_tmp[scm_tmp == 0] = 100
                index = np.unravel_index(np.argmin(scm_tmp), scm_tmp.shape)
                if self.eq_model.family_dict is not None:
                    for el in self.eq_model.family_dict[index[0]]:
                        for el2 in self.eq_model.family_dict[index[1]]:
                            scm[el, el2] = 0
                    # print(f'Remove edge: {index[0]} -> {index[1]}')
                else:
                    # print(f'Remove edge: {index}')
                    scm[index] = 0
                    self.dag = scm.copy()
        scm_tmp = scm.copy()
        if self.eq_model.family_dict is not None:
            rows = []
            for key, value in self.eq_model.family_dict.items():
                rows += [np.expand_dims(scm[value[0], :], axis=0)]
            rows = np.concatenate(rows, axis=0)
            cols = []
            for key, value in self.eq_model.family_dict.items():
                cols += [np.expand_dims(rows[:, value[0]], axis=1)]
            scm_tmp = np.concatenate(cols, axis=1)
        self.dag = scm_tmp.copy()
        

    def forward(self, x):
        return self.do(x, [], None, train=False)

    def _train_do(self, x, intervention_idxs, c, train=True, return_intervened=True, sample=False):
        c_emb_dict = []
        c_context_dict = {}
        c_emb_true_dict = [0 for el in range(self.n_symbols)]
        s_prior = [0 for el in range(self.n_symbols)]
        if train:
            context_posterior_list = []
            for current_node in range(self.n_symbols):
                # compute the context (exogenous variable) of each concept from the input embedding
                if self.probabilistic:
                    context_mean = self.concept_context_generators_mean[current_node](x)
                    context_var = self.concept_context_generators_var[current_node](x)
                    context_sigma = torch.exp(context_var / 2) + EPS
                    context_posterior_dist  = torch.distributions.Normal(context_mean, context_sigma)
                    context = context_posterior_dist.rsample()
                    context_posterior_list.append(context_posterior_dist)
                else:
                    context = self.concept_context_generators[current_node](x)
                c_emb_true = self._build_concept_embedding(context, c[:, current_node].unsqueeze(1))
                c_emb_true_dict[current_node] = c_emb_true
                c_context_dict[current_node] = context
                s_prior[current_node] = self.concept_prob_predictor[current_node](c_context_dict[current_node])
            s_prior = torch.cat(s_prior, axis=1)
            if self.eq_model.family_dict is not None:
                s_prior[:, :self.n_concepts] = torch.sigmoid(s_prior[:, :self.n_concepts])
                s_prior[:, self.n_concepts:] = torch.softmax(s_prior[:, self.n_concepts:], dim=1)
            else: 
                s_prior = torch.sigmoid(s_prior)
            s_emb_prior = torch.stack(c_emb_true_dict).permute(1, 0, 2)
            s_pred_list = [0 for el in range(self.n_symbols)]
            c_emb_post_dict = [0 for el in range(self.n_symbols)]
            context_posterior_dict = [0 for el in range(self.n_symbols)]
            for current_node in range(self.n_symbols):
                context_posterior = self.eq_model(s_emb_prior.permute(0, 2, 1))[:, current_node]
                context_posterior_dict[current_node] = context_posterior
                s_pred = self.concept_prob_predictor[current_node](context_posterior)
                s_pred_list[current_node] = s_pred
            s_pred_list = torch.cat(s_pred_list, axis=1)
            if self.eq_model.family_dict is not None:
                for key, value in self.eq_model.family_dict.items():
                    s_pred_list[:, :self.n_concepts] = torch.sigmoid(s_pred_list[:, :self.n_concepts])
                    s_pred_list[:, self.n_concepts:] = torch.softmax(s_pred_list[:, self.n_concepts:], dim=1)
            else:
                s_pred_list = torch.sigmoid(s_pred_list)
            for current_node in range(self.n_symbols):
                c_emb_true = self._build_concept_embedding(context_posterior_dict[current_node], s_pred_list[:, current_node].unsqueeze(1))
                c_emb_post_dict[current_node] = c_emb_true
            s_emb_post = torch.stack(c_emb_post_dict).permute(1, 0, 2)
            s_preds = s_pred_list
        else:
            self.compute_parent_indices()
            is_root = self.scm.sum(dim=0) == 0
            s_pred_list = [0 for el in range(self.n_symbols)]
            context_posterior_list = []
            s_preds_tmp = []
            context_posterior_list = []
            concept_group = []
            for current_node, parent_nodes in self.parent_indices.items():
                # compute the context (exogenous variable) of each concept from the input embedding
                if self.probabilistic:
                    context_mean = self.concept_context_generators_mean[current_node](x)
                    if sample:
                        context_var = self.concept_context_generators_var[current_node](x)
                        context_sigma = torch.exp(context_var / 2) + EPS
                        context_posterior_dist  = torch.distributions.Normal(context_mean, context_sigma)
                        context = context_posterior_dist.rsample()
                        context_posterior_list.append(context_posterior_dist)
                    else:
                        context = context_mean
                else:
                    context = self.concept_context_generators[current_node](x)
                c_context_dict[current_node] = context
                if is_root[current_node]:
                    s_pred = self.concept_prob_predictor[current_node](context)
                else:
                    context_posterior = self.eq_model(c_emb_dict.permute(0, 2, 1))[:, current_node]
                    s_pred = self.concept_prob_predictor[current_node](context_posterior)
                if self.eq_model.family_dict is None or current_node < self.n_concepts:
                    s_pred = torch.sigmoid(s_pred)
                # Time to check for interventions
                c_int = self._after_interventions(
                    prob=s_pred,
                    concept_idx=current_node,
                    intervention_idxs=intervention_idxs,
                    c_true=c,
                )

                # during training: use concept ground truth values to compute the concept embeddings
                # but return predictions to get the gradient
                if train:
                    c_to_use = c_int
                    s_pred_list[current_node] = (s_pred)
                # at test time during interventions
                else:
                    # if intervention applies to current concept: use ground truth values
                    if current_node in intervention_idxs:
                        c_to_use = c_int
                        s_pred_list[current_node] = (c_int)
                    # else: use predictions
                    else:
                        c_to_use = s_pred
                        s_pred_list[current_node] = (s_pred)
                context_posterior_list.append(context)
                if self.eq_model.family_dict is not None and current_node >= self.n_concepts:
                    concept_group.append(current_node)
                    s_preds_tmp.append(c_to_use)
                    for key, value in self.eq_model.family_dict.items():
                        if concept_group == value:
                            concept_group = []
                            s_preds_tmp = torch.cat(s_preds_tmp, axis=1)
                            s_preds_tmp = torch.softmax(s_preds_tmp, dim=1)
                            for i, el in enumerate(value):
                                s_pred_list[el] = s_preds_tmp[:, i].unsqueeze(1)
                                c_emb_true = self._build_concept_embedding(context, c_to_use)
                                if c_emb_dict == []:
                                    c_emb_dict = torch.zeros(c_emb_true.shape[0], self.n_symbols, c_emb_true.shape[1])
                                c_emb_dict[:, el] = c_emb_true
                            s_preds_tmp = []
                else:
                    c_emb_true = self._build_concept_embedding(context, c_to_use)
                    if c_emb_dict == []:
                        c_emb_dict = torch.zeros(c_emb_true.shape[0], self.n_symbols, c_emb_true.shape[1])
                    c_emb_dict[:, current_node] = c_emb_true
            s_prior = torch.cat(s_pred_list, axis=1)
            s_preds = torch.cat(s_pred_list, axis=1)
            s_emb_prior = c_emb_dict
            s_emb_post = c_emb_dict
        c_preds = s_preds[:, :self.n_concepts]
        y_preds = s_preds[:, self.n_concepts:]
        
        return s_prior, s_preds, s_emb_prior, s_emb_post, context_posterior_list
    

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return c_true[:, concept_idx:concept_idx + 1] 


    def do(self, x, intervention_idxs, c, train=False):
        if train:

            if self.eq_model.family_dict is not None:
                int_list = list(range(len(self.eq_model.family_dict.keys())))
                n = 1
                selected_key = random.sample(int_list, n)[0]
                selected_elements = torch.tensor(self.eq_model.family_dict[selected_key])
                I = torch.eye(len(self.eq_model.family_dict[selected_key]))
                # random index to select a row in I 
                idx1 = torch.randint(0, I.shape[0], (x.shape[0],))
                # another random index to select a row in I except idx1
                idx2 = torch.randint(0, I.shape[0], (x.shape[0],))
                filter_wrong_idx = idx2 == idx1
                idx2[filter_wrong_idx] = (idx2[filter_wrong_idx] + 1) % I.shape[0]
                c0 = (torch.randn((1, self.n_symbols)) > 0.5).float().repeat(x.shape[0], 1)
                c0[:, selected_elements] = I[idx1]
                c1 = (torch.randn((1, self.n_symbols)) > 0.5).float().repeat(x.shape[0], 1)
                c1[:, selected_elements] = I[idx2]
            else:
                int_list = list(range(self.n_concepts))  # Replace this with your list of integers

                # The number of elements you want to select
                n = 1 # random.randint(0, self.n_concepts)  # Replace this with the number of elements you want to select

                # Selecting n random elements from the list
                selected_elements = torch.tensor(random.sample(int_list, n))

                c0 = (torch.randn((1, self.n_symbols)) > 0.5).float().repeat(x.shape[0], 1)
                c1 = 1 - c0

            s_preds_prior0, s_preds_posterior0, _, _, _ = self._train_do(x, selected_elements, c0, train=True, return_intervened=True)
            s_preds_prior1, s_preds_posterior1, _, _, _ = self._train_do(x, selected_elements, c1, train=True, return_intervened=True)

            int_idx = torch.arange(self.n_concepts)
            s_preds_prior, s_preds_posterior, s_emb_prior, s_emb_posterior, context_posterior = self._train_do(x, intervention_idxs=int_idx, c=c, train=True, return_intervened=False)

            c_preds_prior = s_preds_prior[:, :self.n_concepts]
            y_preds_prior = s_preds_prior[:, self.n_concepts:]
            c_preds_posterior = s_preds_posterior[:, :self.n_concepts]
            y_preds_posterior = s_preds_posterior[:, self.n_concepts:]
            return c_preds_prior, y_preds_prior, c_preds_posterior, y_preds_posterior, s_preds_posterior0, s_preds_posterior1, s_emb_prior, s_emb_posterior, context_posterior

        else:
            # after training we need to compute the DAG and use it for inference
            # all test-time operations will be done starting from the root nodes of the DAG moving forward to the leaves
            self.compute_parent_indices()
            is_root = self.scm.sum(dim=0) == 0
            assert self.parent_indices is not None, "Parent indices not found. Please train DAGMA MLP first."
            assert sum(is_root) > 0, "No root nodes found. Please train DAGMA MLP first."


            s_preds_prior, s_preds_posterior, s_emb_prior, s_emb_posterior, _ = self._train_do(x, intervention_idxs=intervention_idxs, c=c, train=False, return_intervened=True)

            c_preds = s_preds_posterior[:, :self.n_concepts]
            y_preds = s_preds_posterior[:, self.n_concepts:]

            return c_preds, y_preds

    def _explain(self, c_pred, current_node, parent_nodes, c_pred_list, feature_names=None):
        explanation = f'c_{current_node} \\leftarrow '
        if len(parent_nodes) == 0:
            return f'${explanation} \\epsilon_{current_node}$', None
        else:
            parent_preds = torch.cat([c_pred_list[parent] for parent in parent_nodes], dim=1)
            preds_table = torch.cat([parent_preds, c_pred], dim=1) > 0.5
            # truth_table = torch.unique(preds_table > 0.5, dim=0)
            unique_parts = preds_table[:, :-1]
            last_elements = preds_table[:, -1]

            unique_matrix, count = torch.unique(preds_table, dim=0, return_counts=True)
            final_unique_rows = []
            for r in unique_matrix:
                filter_r = (unique_matrix[:, :-1] == r[:-1]).all(dim=-1)
                filtered_matrix = unique_matrix[filter_r]
                if filtered_matrix.shape[0] == 1:
                    final_unique_rows.append(r)
                else:
                    # check the most common ones
                    max_index = count[filter_r].argmax()
                    final_unique_rows.append(unique_matrix[filter_r][max_index].squeeze(0))
            final_unique_rows = torch.stack(final_unique_rows)
            truth_table = torch.unique(final_unique_rows, dim=0)                    
            active_mask = truth_table[:, -1] == True
            active_parents_table = truth_table[active_mask, :-1].int().numpy().tolist()
            not_active_parents_table = truth_table[~active_mask, :-1].int().numpy().tolist()
            
            n = parent_preds.shape[-1]
            symbols_txt = ' '.join([f'c_{parent}' for parent in parent_nodes])
            symbols_list = symbols(symbols_txt)
            if len(active_parents_table) == 0:
                exp = false
            elif len(not_active_parents_table) == 0:
                exp = true
            else:
                to_remove_matrix = np.concatenate((active_parents_table, not_active_parents_table), axis=0)
                all_combinations = np.array([[(i >> j) & 1 for j in range(n-1, -1, -1)] for i in range(2**n)])
                filtered_binary_matrix = [row.tolist() for row in all_combinations if not any(np.array_equal(row, remove) for remove in to_remove_matrix)]
                try:
                    _ = iter(symbols_list)
                except:
                    symbols_list = [symbols_list]
                exp = SOPform(symbols_list, active_parents_table, filtered_binary_matrix)
            explanation += str(exp)
        return f'${explanation}$', exp
    
    def explain(self, x, c=None, feature_names=None):
        s_emb_list, s_pred_list = [0 for el in range(self.n_symbols)], [0 for el in range(self.n_symbols)]

        # compute concept truth values
        explanations = {}
        explanations_raw = {}
        s_emb_dict = {}
        c_context_dict = {}
        self.compute_parent_indices()
        is_root = self.scm.sum(dim=0) == 0
        c_emb_dict = []
        
        c_preds_prior, y_preds_prior, c_preds_posterior, y_preds_posterior, _, _, _, _, _ = self.do(x, c=c, intervention_idxs=torch.arange(c.shape[1]), train=True)
        s_pred_pre = torch.cat((c_preds_prior, y_preds_prior), dim=-1)
        s_pred_post = torch.cat((c_preds_posterior, y_preds_posterior), dim=-1)
        s_preds = s_pred_pre.clone()
        s_preds[:, ~is_root] = s_pred_post[:, ~is_root]
        s_preds = [s.unsqueeze(1) for s in s_preds.T]
        c_true_list = [c.unsqueeze(1) for c in c.T]
        for current_node, parent_nodes in self.parent_indices.items():
            explanations[current_node], explanations_raw[current_node] = self._explain(s_preds[current_node], current_node, parent_nodes, c_true_list, feature_names=feature_names)

        return explanations, explanations_raw

