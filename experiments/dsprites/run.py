import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from causal_cbm.utils import cace_score, explanation_to_latex, evaluate_exp, interventions_from_root, compute_pns_matrix, interventions_flatten, conditional_entropy_dag
import numpy as np
import os
from model import CEM, CBM, CausalHCEM, StandardE2E
import pytorch_lightning as pl
from experiments.dsprites.dataset import load_preprocessed_data
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = ''

seed_everything(0)

n_seeds = 5
n_samples = 800
embedding_size = 8
ce_size = 5
gamma = 1
label_names = ['Shape', 'Size', 'PosY', 'PosX', 'Color', 'Label']
results_dir = './results'
index_perturb = 3
index_block = 1
os.makedirs(results_dir, exist_ok=True)

dag_init = torch.FloatTensor([[0, 1, 0, 0, 1, 0], 
                            [0, 0, 0, 0, 0, 1], 
                            [0, 0, 0, 0, 1, 0], 
                            [0, 1, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0]
                            ])

results = []
interventions = []
explanations = []
pns_list = []
for i in range(n_seeds):
    seed = i + 1

    lambda_cace_2 = 0.05

    x_train, c_train, y_train, x_test, c_test, y_test = load_preprocessed_data(base_dir='../../datasets/dsprites')
    n = x_train.shape[0]
    x_val = x_train[-int(n*0.2):]
    c_val = c_train[-int(n*0.2):]
    y_val = y_train[-int(n*0.2):]
    x_train = x_train[:int(n*0.8)]
    c_train = c_train[:int(n*0.8)]
    y_train = y_train[:int(n*0.8)]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]
    s_train = torch.cat((c_train, y_train), dim=1)
    s_val = torch.cat((c_val, y_val), dim=1)
    s_test = torch.cat((c_test, y_test), dim=1)
    
    train_loader = DataLoader(TensorDataset(x_train, s_train), batch_size=128, shuffle=True)
    train_loader_cbm = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, s_val), batch_size=128, shuffle=False)
    val_loader_cbm = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=128, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, s_test), batch_size=128, shuffle=False)
    test_loader_cbm = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=128, shuffle=False)
    n_symbols = s_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]
    tp_size = n_concepts*ce_size

    models = {

        'CausalHCEM_0_0.05_det': CausalHCEM(x_train.shape[1], embedding_size, n_concepts, n_classes, ce_size, gamma, 0, lambda_cace_2, probabilistic=False),
        'CausalHCEM_0_0.05_det_given': CausalHCEM(x_train.shape[1], embedding_size, n_concepts, n_classes, ce_size, gamma, 0, lambda_cace_2, probabilistic=False),
        'CEM': CEM(x_train.shape[1], embedding_size, n_concepts, n_classes, ce_size, tp_size),
        'CBM': CBM(x_train.shape[1], embedding_size, n_concepts, n_classes),
        # 'BB': StandardE2E(x_train.shape[1], n_classes+n_concepts, emb_size=embedding_size),
    }

    for model_name, model in models.items():
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
        print(f'Training {model_name} with seed {seed}/{n_seeds}...')

        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_concept_accuracy", mode="max", save_weights_only=True)
        if model_name.endswith('given'):
            model.concept_embedder.eq_model.fc1.weight = torch.nn.Parameter(dag_init, requires_grad=False)
        elif model_name not in ['CEM', 'CBM', 'BB']:
            cov = torch.abs(torch.tensor(np.corrcoef(s_train.T))).float()/2
            # cov = conditional_entropy_dag(s_train)
            # cov = torch.tensor(cov).float()
            with torch.no_grad():
                print(model.concept_embedder.eq_model.fc1.weight)
                print(cov)
                model.concept_embedder.eq_model.fc1.weight = torch.nn.Parameter(cov, requires_grad=True)
        # try:
        trainer = Trainer(max_epochs=200, accelerator='cpu', enable_checkpointing=True, callbacks=checkpoint_callback)
        trainer.fit(model, train_loader_cbm, val_loader_cbm)
        # except:
        #     continue
        model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
        model.eval()

        # data we need to test models
        c_test = s_test[:, :-1].clone()
        y_test = s_test[:, -1]
        s_fake_1 = s_test.clone()
        s_fake_0 = s_test.clone()
        s_fake_1[:, [index_perturb]] = 1
        s_fake_0[:, [index_perturb]] = 0
        c_fake_1 = s_fake_1[:, :-1]
        c_fake_0 = s_fake_0[:, :-1]
        s_perturb = s_test.clone()
        s_perturb[:, [index_perturb]] = 1 - s_test[:, [index_perturb]]
        c_perturb = s_perturb[:, :-1]

        explanation = None
        if model.__class__.__name__ in ['CausalCEM', 'CausalCBM', 'CausalCBMDAGMA', 'CausalCEMDAGMA', 'CausalHCEM']:
            # plot DAG
            # if model.__class__.__name__ == 'CausalCBM':
            model.concept_embedder.compute_parent_indices()
            dag = model.concept_embedder.dag
            # dag = model.concept_embedder.eq_model.fc1_to_adj().detach().cpu().numpy()
            plt.figure(figsize=(4, 4))
            plt.title(f'dSprites DAG')
            dag_tmp = (dag > 0.01).astype(float)
            sns.heatmap(dag_tmp, xticklabels=label_names, yticklabels=label_names, cmap='coolwarm', cbar=True, vmin=0, vmax=1)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'dsprites_heatmap_{model_name}_{seed}.pdf'))
            # plt.show()

            emb = model.encoder(x_train)
            s_pred = model(x_test)
            explanation, explanation_raw = model.concept_embedder.explain(emb, feature_names=label_names, c=s_train)

            fidelity = evaluate_exp(explanation_raw, s_pred)
            print('Fidelity per concept:', fidelity)
            fidelity = np.mean(fidelity)
            print('Fidelity:', np.mean(fidelity))

            print(s_pred.shape, s_train.shape)
            latex_explanation = explanation_to_latex(explanation)
            explanations.append([model_name, seed, latex_explanation])
            pd.DataFrame(explanations, columns=['model', 'seed', 'explanation']).to_csv('explanations_int.csv', index=False)
            # compute standard accuracy
            # do interventions without blocking
            s_pred_c1 = model.forward(x_test, c=s_fake_1, intervention_idxs=[index_perturb])
            s_pred_c0 = model.forward(x_test, c=s_fake_0, intervention_idxs=[index_perturb])

            # do interventions with blocking
            s_pred_c1_block = model.forward(x_test, c=s_fake_1, intervention_idxs=[index_perturb, index_block])
            s_pred_c0_block = model.forward(x_test, c=s_fake_0, intervention_idxs=[index_perturb, index_block])

            # do perturbation without blocking
            s_pred_perturb = model.forward(x_test, c=s_perturb, intervention_idxs=[index_perturb])

            # do perturbation with blocking
            s_pred_perturb_block = model.forward(x_test, c=s_perturb, intervention_idxs=[index_perturb, index_block])


            s_pred = model.forward(x_test)

            # average cace 
            sum_cace = []
            for i in range(n_symbols):
                s_fake_1_tmp = s_test.clone()
                s_fake_0_tmp = s_test.clone()
                s_fake_1_tmp[:, [i]] = 1
                s_fake_0_tmp[:, [i]] = 0
                s_pred_c1_tmp = model.forward(x_test, c=s_fake_1_tmp, intervention_idxs=[i])
                s_pred_c0_tmp = model.forward(x_test, c=s_fake_0_tmp, intervention_idxs=[i])
                filter_out_i = [True if j != i else False for j in range(n_symbols)]
                filter_out_i = torch.tensor(filter_out_i)
                cace = cace_score(s_pred_c1_tmp[:, filter_out_i], s_pred_c0_tmp[:, filter_out_i]).detach().mean().item()
                sum_cace += [cace]
            sum_cace = np.array(sum_cace)
            min_cace = np.min(sum_cace)
            average_cace = np.mean(sum_cace)
            max_cace = np.max(sum_cace)
            print(f'{model_name} average cace: {average_cace}')
            diff_c = torch.abs(s_test[:, :-1] - s_pred[:, :-1])
            diff_c_mean = diff_c.mean(dim=0)
            order_diff = torch.argsort(diff_c_mean, descending=True)
            x_test_shuffled = x_test.clone()
            indexes = torch.randperm(x_test_shuffled.size(0))
            x_test_shuffled = x_test_shuffled[indexes]
            acc_int = interventions_from_root(torch.tensor(dag), model, x_test, s_test, exclude=[n_concepts])
            print(acc_int)

            # PS and PN
            PNS = compute_pns_matrix(x_test, model, dag)
            print(PNS)
        elif model_name in ['CEM', 'CBM']:
            explanation = {'': ''}
            fidelity = 0

            # compute standard accuracy
            s_pred = model.forward(x_test)

            sum_cace = []
            for i in range(n_symbols):
                s_fake_1 = s_test.clone()
                s_fake_0 = s_test.clone()
                s_fake_1[:, [i]] = 1
                s_fake_0[:, [i]] = 0
                s_pred_c1 = model.forward(x_test, c=s_fake_1, intervention_idxs=[i], train=False)
                s_pred_c0 = model.forward(x_test, c=s_fake_0, intervention_idxs=[i], train=False)
                filter_out_i = [True if j != i else False for j in range(n_symbols)]
                filter_out_i = torch.tensor(filter_out_i)
                cace = cace_score(s_pred_c1[:, filter_out_i], s_pred_c0[:, filter_out_i]).detach().mean().item()
                sum_cace += [cace]

            sum_cace = np.array(sum_cace)
            min_cace = np.min(sum_cace)
            average_cace = np.mean(sum_cace)
            max_cace = np.max(sum_cace)
            print(f'{model_name} average cace: {average_cace}')

            x_test_shuffled = x_test.clone()
            indexes = torch.randperm(x_test_shuffled.size(0))
            x_test_shuffled = x_test_shuffled[indexes]
            diff_c = torch.abs(s_test[:, :-1] - s_pred[:, :-1])
            diff_c_mean = diff_c.mean(dim=0)
            order_diff = torch.argsort(diff_c_mean, descending=True)
            acc_int = interventions_from_root(torch.tensor(dag_init), model, x_test, s_test, order_diff, exclude=[n_concepts])
            print(acc_int)
            dag = torch.zeros(n_symbols, n_symbols)
            dag[:, -1] = 1
            dag[-1, -1] = 0
            PNS = compute_pns_matrix(x_test, model, dag.numpy())
            print(PNS)

            # do interventions without blocking
            s_pred_c1 = model.forward(x_test, c=c_fake_1, intervention_idxs=[index_perturb], train=False)
            s_pred_c0 = model.forward(x_test, c=c_fake_0, intervention_idxs=[index_perturb], train=False)

            # do interventions with blocking
            s_pred_c1_block = model.forward(x_test, c=c_fake_1, intervention_idxs=[index_perturb, index_block], train=False)
            s_pred_c0_block = model.forward(x_test, c=c_fake_0, intervention_idxs=[index_perturb, index_block], train=False)

            # do perturbation without blocking
            s_pred_perturb = model.forward(x_test, c=c_perturb, intervention_idxs=[index_perturb], train=False)

            # do perturbation with blocking
            s_pred_perturb_block = model.forward(x_test, c=c_perturb, intervention_idxs=[index_perturb, index_block], train=False)


        # compute metrics
        # s_pred = torch.cat([c_pred, y_pred], dim=1)
        # s_test = torch.cat([c_test, y_test], dim=1)
        if model_name == 'BB':
            s_pred = model.forward(x_test)
            s_accuracy = accuracy_score(s_test.ravel(), s_pred.ravel() > 0)
            print(f'{model_name} label accuracy: {s_accuracy}')
            # save results
            results.append(['dsprites_dataset', model_name, seed, s_accuracy])
            metric_cols = ['label_accuracy']
            df = pd.DataFrame(results, columns=['dataset', 'model', 'seed'] + metric_cols)
            df.to_csv(os.path.join(results_dir, 'results_raw.csv'), index=False)
            df = df.drop(columns=['dataset'])
            mean_scaled = df.groupby('model').agg(lambda x: np.mean(x)) * 100
            variance_scaled = df.groupby('model').agg(lambda x: np.std(x)) * 100
            formatted_dfs = []
            for model in mean_scaled.index:
                formatted_dfs.append(pd.DataFrame({col: f"${mean_scaled.loc[model, col]:.2f} \pm {variance_scaled.loc[model, col]:.2f}$" for col in metric_cols}, index=[model]))
            formatted_df = pd.concat(formatted_dfs)
            print(formatted_df.to_string(index=True, float_format='%.2f'))
            formatted_df.to_csv(os.path.join(results_dir, 'intv_results_table.csv'))
        else:
            s_accuracy = accuracy_score(s_test.ravel(), s_pred.ravel() > 0.5)
            print(f'{model_name} label accuracy: {s_accuracy}')

            cace = cace_score(s_pred_c1[:, -1], s_pred_c0[:, -1]).detach().item()
            cace_block = cace_score(s_pred_c1_block[:, -1], s_pred_c0_block[:, -1]).detach().item()
            label_accuracy_perturb = accuracy_score(s_test[:, -1], s_pred_perturb[:, -1] > 0.5)
            label_accuracy_perturb_block = accuracy_score(s_test[:, -1], s_pred_perturb_block[:, -1] > 0.5)

            # save results
            results.append(['dsprites_dataset', model_name, seed, s_accuracy, average_cace, min_cace, max_cace, fidelity, label_accuracy_perturb, label_accuracy_perturb_block, cace, cace_block])
            metric_cols = ['label_accuracy', 'average_cace', 'min_cace', 'max_cace', 'fidelity', 'label_accuracy_perturb', 'label_accuracy_perturb_block', 'cace', 'cace_block']
            df = pd.DataFrame(results, columns=['dataset', 'model', 'seed'] + metric_cols)
            df.to_csv(os.path.join(results_dir, 'results_raw.csv'), index=False)
            
            df = df.drop(columns=['dataset'])
            mean_scaled = df.groupby('model').agg(lambda x: np.mean(x)) * 100
            variance_scaled = df.groupby('model').agg(lambda x: np.std(x)) * 100
            formatted_dfs = []
            for model in mean_scaled.index:
                formatted_dfs.append(pd.DataFrame({col: f"${mean_scaled.loc[model, col]:.2f} \pm {variance_scaled.loc[model, col]:.2f}$" for col in metric_cols}, index=[model]))
            formatted_df = pd.concat(formatted_dfs)
            print(formatted_df.to_string(index=True, float_format='%.2f'))
            formatted_df.to_csv(os.path.join(results_dir, 'intv_results_table.csv'))
            
            interventions.append(['dsprites_dataset', model_name, seed, acc_int])
            metric_cols = ['acc_int']
            df = pd.DataFrame(interventions, columns=['dataset', 'model', 'seed'] + metric_cols)
            df.to_csv(os.path.join(results_dir, 'interventions.csv'), index=False)

            pns_list.append(['dsprites_dataset', model_name, seed, PNS])
            metric_cols = ['PNS']
            df = pd.DataFrame(pns_list, columns=['dataset', 'model', 'seed'] + metric_cols)
            df.to_csv(os.path.join(results_dir, 'pns.csv'), index=False)