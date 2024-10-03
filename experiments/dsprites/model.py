import torch
from abc import abstractmethod
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from causal_cbm.nn import CausalConceptMessagePassingDAG, ConceptEmbedding, CausalConceptBottleneckMessagePassingDAG, \
    CausalConceptMessagePassingDAGMA, CausalConceptEmbeddingMessagePassing, CausalConceptEmbeddingLayer
from causal_cbm.utils import cace_score, orthogonality_loss

class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")
        self.bce_log = BCEWithLogitsLoss(reduction="mean")

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def _unpack_input(self, I):
        raise NotImplementedError

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        y_preds = self.forward(X)
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(1)
        s = torch.cat([c_true, y_true], dim=1)

        loss = self.bce_log(y_preds.squeeze(), s.float().squeeze())
        task_accuracy = accuracy_score(s.squeeze(), y_preds > 0)
        self.log("train_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(1)
        s = torch.cat([c_true, y_true], dim=1)
        loss = self.bce_log(y_preds.squeeze(), s.float().squeeze())
        task_accuracy = accuracy_score(s.squeeze(), y_preds > 0)
        self.log("val_concept_accuracy", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(1)
        s = torch.cat([c_true, y_true], dim=1)
        loss = self.bce_log(y_preds.squeeze(), s.float().squeeze())
        task_accuracy = accuracy_score(s.squeeze(), y_preds > 0)
        self.log("test_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer



class StandardE2E(NeuralNet):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )

    def _unpack_input(self, I):
        return I[0], I[1], I[2]

    def forward(self, X, explain=False):
        return self.model(X)

class BaseCEM(LightningModule):
    def __init__(self, input_dim, embedding_size):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )
        self.loss = torch.nn.BCELoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)


class CEM(BaseCEM):
    def __init__(self, input_dim, embedding_size, n_concepts, n_classes, ce_size, tp_size):
        super().__init__(input_dim, embedding_size)
        self.concept_embedder = ConceptEmbedding(embedding_size, n_concepts, ce_size)
        self.task_predictor = torch.nn.Sequential(
            torch.nn.Linear(tp_size, tp_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(tp_size, n_classes),
            torch.nn.Sigmoid()
        )
        self.n_symbols = n_concepts + n_classes

    def forward(self, x, c=None, intervention_idxs=None, train=False):
        emb = self.encoder(x)
        c_emb, c_pred = self.concept_embedder(emb, c=c, intervention_idxs=intervention_idxs, train=train, return_intervened=True)
        y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
        s_pred = torch.cat([c_pred, y_pred.unsqueeze(1)], dim=-1)
        return s_pred

    def training_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train, c_train, train=True)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train, train=False)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class CBM(BaseCEM):
    def __init__(self, input_dim, embedding_size, n_concepts, n_tasks):
        super().__init__(input_dim, embedding_size)
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, n_concepts),
            torch.nn.Sigmoid()
        )
        self.task_predictor = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, n_tasks),
            torch.nn.Sigmoid()
        )
        self.n_symbols = n_concepts + n_tasks  

    def forward(self, x, c=None, intervention_idxs=[], train=False):
        emb = self.encoder(x)
        c_pred = self.concept_predictor(emb)
        for idx in intervention_idxs:
            if idx < c_pred.shape[-1]:
                c_pred[:, idx] = c[:, idx]
        y_pred = self.task_predictor(c_pred)
        s_pred = torch.cat([c_pred, y_pred], dim=-1)
        return s_pred

    def training_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train, train=False)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class CausalHCEM(BaseCEM):
    def __init__(self, input_dim, embedding_size, n_concepts, n_classes, ce_size, gamma, lambda_orth=0, lambda_cace=0, probabilistic=False, root_loss_l = 0.1,
                 weight=None, family_of_concepts=None):
        super().__init__(input_dim, embedding_size)
        self.concept_embedder = CausalConceptEmbeddingLayer(embedding_size, n_concepts, n_classes, embedding_size, gamma=gamma, probabilistic=probabilistic)
        self.mse_loss = torch.nn.MSELoss()
        self.n_symbols = n_concepts + n_classes 
        self.lambda_orth = lambda_orth
        self.lambda_cace = lambda_cace
        self.root_loss_l = root_loss_l

    def forward(self, x, c=None, intervention_idxs=[]):
        emb = self.encoder(x)
        c_pred, y_pred = self.concept_embedder.do(emb, c=c, intervention_idxs=intervention_idxs, train=False)
        s_pred = torch.cat([c_pred, y_pred], dim=-1)
        return s_pred

    def training_step(self, batch, batch_idx):
        x, c, y = batch
        emb = self.encoder(x)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        c_tmp = torch.cat([c, y], dim=-1)
        (c_preds_prior, y_preds_prior,
         c_preds_posterior, y_preds_posterior,
         s_preds_posterior0, s_preds_posterior1, s_emb_prior, s_emb_posterior, context_post) = self.concept_embedder.do(emb, c=c_tmp, intervention_idxs=torch.arange(c_tmp.shape[1]), train=True)
        # reconstr_loss = self.mse_loss(s_emb_prior, s_emb_posterior.detach())
        cace_loss = cace_score(s_preds_posterior1, s_preds_posterior0).norm()
        # cace_loss = 0
        self.concept_embedder.compute_parent_indices()

        is_root = self.concept_embedder.scm.sum(dim=0) == 0
        emb_used = torch.cat([s_emb_prior[:, is_root], s_emb_posterior[:, ~is_root]], dim=1)
        orth_loss = orthogonality_loss(emb_used) 
        dag_loss = self.concept_embedder.eq_model.h_func() # + self.root_loss_l*self.concept_embedder.eq_model.root_loss()
        prior_loss = self.loss(c_preds_prior, c) + self.loss(y_preds_prior.squeeze(), y.squeeze())
        posterior_loss = self.loss(c_preds_posterior, c) + self.loss(y_preds_posterior.squeeze(), y.squeeze())
        # loss = prior_loss + posterior_loss + reconstr_loss + dag_loss # + 0.05 / (cace_loss + 1e-6)
        loss = prior_loss + posterior_loss + dag_loss + self.lambda_orth*orth_loss + self.lambda_cace / (cace_loss + 1e-6)
        if self.concept_embedder.probabilistic:
            for i, el in enumerate(self.concept_embedder.prior):
                loss += 0.08*torch.distributions.kl_divergence(el, context_post[i]).mean()
        prior_task_accuracy = accuracy_score(y.cpu().squeeze(), y_preds_prior.detach().cpu().squeeze() > 0.5)
        prior_concept_accuracy = accuracy_score(c.cpu(), c_preds_prior.detach().cpu() > 0.5)
        
        posterior_task_accuracy = accuracy_score(y.cpu().squeeze(), y_preds_posterior.detach().cpu().squeeze() > 0.5)
        posterior_concept_accuracy = accuracy_score(c.cpu(), c_preds_posterior.detach().cpu() > 0.5)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('dag', dag_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('reconstr', reconstr_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('cace', cace_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('c acc pri', prior_concept_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('c acc pos', posterior_concept_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('y acc pri', prior_task_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('y acc pos', posterior_task_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        s = torch.cat([c, y], dim=-1)
        emb = self.encoder(x)
        (c_preds, y_preds) = self.concept_embedder(emb)
        s_pred = torch.cat((c_preds, y_preds), dim=-1)
        loss = self.loss(s_pred, s) 
        concept_accuracy = accuracy_score(s.cpu().ravel(), s_pred.detach().cpu().ravel() > 0.5)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_concept_accuracy', concept_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    

