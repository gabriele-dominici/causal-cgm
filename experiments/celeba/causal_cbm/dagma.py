import torch
import torch.nn as nn
import numpy as np
from torch import optim
import copy
from tqdm.auto import tqdm
import typing
import torch.nn.functional as F

def threshold_relu(x, th):
    return torch.maximum(x - th, torch.zeros_like(x))

class DagmaCE(nn.Module):
    """
    Class that models the structural equations for the causal graph using MLPs.
    """

    def __init__(self, n_concepts: int, n_classes: int, dims: typing.List[int], bias: bool = False, gamma: float = 10.0):
        r"""
        Parameters
        ----------
        n_concepts : typing.List[int]
            Number of concepts in the dataset.
        n_classes : typing.List[int]
            Number of classes in the dataset.
        dims : typing.List[int]
            Number of neurons in hidden layers of each MLP representing each structural equation.
        bias : bool, optional
            Flag whether to consider bias or not, by default ``True``
        """
        super(DagmaCE, self).__init__()
        assert len(dims) >= 2
        # assert dims[-1] == 1
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_symbols = n_concepts + n_classes
        self.dims = dims
        self.I = torch.eye(self.n_symbols)
        self.mask = torch.ones(self.n_symbols, self.n_symbols)
        self.mask = self.mask - self.I
        self.mask[n_concepts:] = torch.zeros(n_classes, self.n_symbols)

        self.edges_to_check = []
        self.edge_matrix = torch.nn.Parameter(torch.zeros(self.n_symbols, self.n_symbols))
        self.family_dict = None

        self.family_of_concepts = None

        # threshold for the adjacency matrix as learnable parameter
        self.th = nn.Parameter(torch.tensor([0.1]))
        # self.gamma = nn.Parameter(torch.tensor([10.0]))
        self.gamma = gamma
        self.fc1 = nn.Linear(self.n_symbols, self.n_symbols, bias=bias)
        # nn.init.zeros_(self.fc1.weight)
        # if bias:
        #     nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for lid, l in enumerate(range(len(dims) - 1)):
            layers.append(nn.Linear(dims[l], dims[l + 1], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def edge_to_check(self, edge, family_of_concepts=None):
        self.add_edges = []
        self.edges_to_check = edge
        for _ in edge:
            self.add_edges += [nn.Parameter(torch.tensor([0.0]))]
        if family_of_concepts is not None:
            self.set_family_of_concepts(family_of_concepts)

    def set_family_of_concepts(self, family_of_concepts):
        self.family_of_concepts = family_of_concepts
        self.family_dict = {}
        count = 0
        for i, el in enumerate(family_of_concepts):
            self.family_dict[i] = list(range(count, count + el))
            count += el

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        r"""
        Applies the current states of the structural equations to the dataset X

        Parameters
        ----------
        x : torch.Tensor
            Input dataset with shape :math:`(n,d)`.

        Returns
        -------
        torch.Tensor
            Result of applying the structural equations to the input data.
            Shape :math:`(n,d)`.
        """
        fc1_weight = self.fc1_to_adj()
        x = torch.matmul(x, fc1_weight)
        x = x.permute(0, 2, 1)
        for fc in self.fc2:
            x = F.leaky_relu(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        fc1_weight = self.fc1_to_adj()
        A = torch.abs(fc1_weight)  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.n_symbols * np.log(s)
        return torch.abs(h)
    
    def root_loss(self):
        fc1_weight = self.fc1_to_adj()
        n_edges = fc1_weight[:self.n_concepts, :self.n_concepts].sum()
        n_y = fc1_weight[:, self.n_concepts:].sum()
        return 1/(n_edges + 1)#  + (n_y-1)**2  

    def fc1_l1_reg(self) -> torch.Tensor:
        r"""
        Takes L1 norm of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the L1 norm of first FC layer.
        """
        fc1_weight = self.fc1_to_adj()
        return torch.sum(torch.abs(fc1_weight), dim=0).norm(p=0)

    def fc1_smoothl0_reg(self) -> torch.Tensor:
        r"""
        Takes smoothed L0 norm of the weights in the first fully-connected layer

        Yang, Cuili, et al. "Design of Extreme Learning Machine with Smoothed ℓ 0 Regularization." Mobile Networks and Applications 25 (2020): 2434-2446.

        Returns
        -------
        torch.Tensor
            A scalar value of the smoothed L0 norm of first FC layer.
        """
        fc1_weight = self.fc1_to_adj()

        cols_w = torch.sum(torch.abs(fc1_weight), dim=0)
        cols_sl0 = 1 - torch.div(torch.sin(cols_w / self.gamma), cols_w / self.gamma + 1e-6)

        rows_w = torch.sum(torch.abs(fc1_weight), dim=1)
        rows_sl0 = 1 - torch.div(torch.sin(rows_w / self.gamma), rows_w / self.gamma + 1e-6)

        return 1 / (cols_sl0.sum() + 1e-6) + 1 / (rows_sl0.sum() + 1e-6)

    def fc1_entropy_reg(self) -> torch.Tensor:
        r"""
        Takes Entropy of the weights in the first fully-connected layer

        Returns
        -------
        torch.Tensor
            A scalar value of the Entropy of first FC layer.
        """
        fc1_weight = self.fc1_to_adj()
        w = torch.sum(torch.abs(fc1_weight), dim=0) + 1e-6
        return torch.sum(w * torch.log(w))

    def fc1_to_adj(self) -> torch.Tensor:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix
        """
        W = torch.abs(self.fc1.weight * self.mask)
        
        # W_mask = torch.abs(W) / torch.max(torch.abs(W))
        # W = threshold_relu(W_mask, self.th)
        W_mask = torch.abs(W) / torch.max(torch.abs(W)) > self.th
        W = W * W_mask
        mask_tmp = torch.zeros_like(W)
        for i, el in enumerate(self.edges_to_check):
            if self.family_of_concepts is not None:
                index1 = self.family_dict[el[0]][0]
                index2 = self.family_dict[el[1]][0]
                mask_tmp[index1, index2] += 1
            else:
                mask_tmp[el[0], el[1]] += 1
            # W[el[0], el[1]] = (value > 0.5).float().detach() - value.detach() + value
            # W[el[1], el[0]] = ((1 - value) >= 0.5).float().detach() - value.detach() + value
        W = W + mask_tmp * (torch.sigmoid(self.edge_matrix) > 0.5).float().detach() - torch.sigmoid(self.edge_matrix).detach() + torch.sigmoid(self.edge_matrix)
        for i, el in enumerate(self.edges_to_check):
            if self.family_of_concepts is not None:
                index1 = self.family_dict[el[1]][0]
                index2 = self.family_dict[el[0]][0]
                W[index1, index2] += 1 - (W[index2, index1] > 0.5).float()
            else:
                W[el[1], el[0]] += 1 - (W[el[0], el[1]] > 0.5).float()
        # if self.family_of_concepts is not None:
        #     for _, el in enumerate(self.edges_to_check):
        #         for i in range(len(self.family_dict[el[0]])):
        #             index1 = self.family_dict[el[0]][i]
        #             for j in range(len(self.family_dict[el[1]])):
        #                 index2 = self.family_dict[el[1]][j]
        #                 if i == 0 and j == 0:
        #                     continue
        #                 W[index1, index2] += W[self.family_dict[el[0]][0], self.family_dict[el[1]][0]]
        #                 W[index2, index1] += W[self.family_dict[el[1]][0], self.family_dict[el[0]][0]]
        mask_representer = torch.ones_like(W)
        if self.family_of_concepts is not None:
            for _, value in self.family_dict.items():
                idxs = value[1:]
                for idx in idxs:
                    mask_representer[idx, :] = 0
                    mask_representer[:, idx] = 0
            W = W * mask_representer
            if self.family_of_concepts is not None:
                for _, value in self.family_dict.items():
                    idx_to_copy = value[0]
                    for el in value[1:]:
                        W[:, el] += W[:, idx_to_copy]
                        W[el, :] += W[idx_to_copy, :]
        return W # F.relu(W - torch.abs(self.th))

    # def fc1_to_adj(self) -> torch.Tensor:  # [j * m1, i] -> [i, j]
    #     r"""
    #     Computes the induced weighted adjacency matrix W from the first FC weights.
    #     Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

    #     Returns
    #     -------
    #     np.ndarray
    #         :math:`(d,d)` weighted adjacency matrix
    #     """
    #     W = torch.abs(self.fc1.weight * self.mask)
    #     # W_mask = torch.abs(W) / torch.max(torch.abs(W)) > 0.5
    #     W_mask = torch.abs(W) / torch.max(torch.abs(W)) > self.th
    #     W = W * W_mask
    #     mask_tmp = torch.zeros_like(W)
    #     for i, el in enumerate(self.edges_to_check):
    #         if self.family_of_concepts is not None:
    #             index1 = self.family_dict[el[0]][0]
    #             index2 = self.family_dict[el[1]][0]
    #             mask_tmp[index1, index2] += 1
    #             mask_tmp[index2, index1] += 1
    #         else:
    #             mask_tmp[el[0], el[1]] += 1
    #         # W[el[0], el[1]] = (value > 0.5).float().detach() - value.detach() + value
    #         # W[el[1], el[0]] = ((1 - value) >= 0.5).float().detach() - value.detach() + value
    #     print(mask_tmp * torch.sigmoid(self.edge_matrix))
    #     W = W + mask_tmp * (torch.sigmoid(self.edge_matrix) > 0.5).float().detach() - torch.sigmoid(self.edge_matrix).detach() + torch.sigmoid(self.edge_matrix)
    #     if self.family_of_concepts is not None:
    #         for _, el in enumerate(self.edges_to_check):
    #             for i in range(len(self.family_dict[el[0]])):
    #                 index1 = self.family_dict[el[0]][i]
    #                 for j in range(len(self.family_dict[el[1]])):
    #                     index2 = self.family_dict[el[1]][j]
    #                     if i == 0 and j == 0:
    #                         continue
    #                     W[index1, index2] += W[self.family_dict[el[0]][0], self.family_dict[el[1]][0]]
    #                     W[index2, index1] += W[self.family_dict[el[1]][0], self.family_dict[el[0]][0]]
    #     return W # F.relu(W - torch.abs(self.th))


class CausalLayer(DagmaCE):
    """
    Class that models the structural equations for the causal graph using MLPs.
    """

    def __init__(self, n_concepts: int, n_classes: int, dims: typing.List[int], bias: bool = False, gamma: float = 10.0):
        r"""
        Parameters
        ----------
        n_concepts : typing.List[int]
            Number of concepts in the dataset.
        n_classes : typing.List[int]
            Number of classes in the dataset.
        dims : typing.List[int]
            Number of neurons in hidden layers of each MLP representing each structural equation.
        bias : bool, optional
            Flag whether to consider bias or not, by default ``True``
        """
        super().__init__(n_concepts, n_classes, dims, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        r"""
        Applies the current states of the structural equations to the dataset X

        Parameters
        ----------
        x : torch.Tensor
            Input dataset with shape :math:`(n,d)`.

        Returns
        -------
        torch.Tensor
            Result of applying the structural equations to the input data.
            Shape :math:`(n,d)`.
        """
        fc1_weight = self.fc1_to_adj()
        x = torch.matmul(x, fc1_weight)
        x = x.permute(0, 2, 1)
        for fc in self.fc2:
            x = F.leaky_relu(x)
            x = fc(x)
        x = F.leaky_relu(x)
        return x
    

class DagmaNonlinear:
    """
    Class that implements the DAGMA algorithm
    """

    def __init__(self, model: nn.Module, verbose: bool = False, dtype: torch.dtype = torch.double):
        """
        Parameters
        ----------
        model : nn.Module
            Neural net that models the structural equations.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit`. Defaults to ``False``.
        dtype : torch.dtype, optional
            float number precision, by default ``torch.double``.
        """
        self.vprint = print if verbose else lambda *a, **k: None
        self.model = model
        self.dtype = dtype

    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the logarithm of the MSE loss:
            .. math::
                \frac{d}{2} \log\left( \frac{1}{n} \sum_{i=1}^n (\mathrm{output}_i - \mathrm{target}_i)^2 \right)

        Parameters
        ----------
        output : torch.Tensor
            :math:`(n,d)` output of the model
        target : torch.Tensor
            :math:`(n,d)` input dataset

        Returns
        -------
        torch.Tensor
            A scalar value of the loss.
        """
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss

    def minimize(self,
                 max_iter: float,
                 lr: float,
                 lambda1: float,
                 lambda2: float,
                 mu: float,
                 s: float,
                 lr_decay: float = False,
                 tol: float = 1e-6,
                 pbar: typing.Optional[tqdm] = None,
                 ) -> bool:
        r"""
        Solves the optimization problem:
            .. math::
                \arg\min_{W(\Theta) \in \mathbb{W}^s} \mu \cdot Q(\Theta; \mathbf{X}) + h(W(\Theta)),
        where :math:`Q` is the score function, and :math:`W(\Theta)` is the induced weighted adjacency matrix
        from the model parameters.
        This problem is solved via (sub)gradient descent using adam acceleration.

        Parameters
        ----------
        max_iter : float
            Maximum number of (sub)gradient iterations.
        lr : float
            Learning rate.
        lambda1 : float
            L1 penalty coefficient. Only applies to the parameters that induce the weighted adjacency matrix.
        lambda2 : float
            L2 penalty coefficient. Applies to all the model parameters.
        mu : float
            Weights the score function.
        s : float
            Controls the domain of M-matrices.
        lr_decay : float, optional
            If ``True``, an exponential decay scheduling is used. By default ``False``.
        tol : float, optional
            Tolerance to admit convergence. Defaults to 1e-6.
        pbar : tqdm, optional
            Controls bar progress. Defaults to ``tqdm()``.

        Returns
        -------
        bool
            ``True`` if the optimization succeded. This can be ``False`` when at any iteration, the model's adjacency matrix
            got outside of the domain of M-matrices.
        """
        self.vprint(f'\nMinimize s={s} -- lr={lr}')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(.99, .999), weight_decay=mu * lambda2)
        if lr_decay is True:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        obj_prev = 1e16
        for i in range(max_iter):
            optimizer.zero_grad()
            h_val = self.model.h_func(s)
            if h_val.item() < 0:
                self.vprint(f'Found h negative {h_val.item()} at iter {i}')
                return False
            X_hat = self.model(self.X)
            score = self.log_mse_loss(X_hat, self.X)
            l1_reg = lambda1 * self.model.fc1_l1_reg()
            obj = mu * (score + l1_reg) + h_val
            obj.backward()
            optimizer.step()
            if lr_decay and (i + 1) % 1000 == 0:  # every 1000 iters reduce lr
                scheduler.step()
            if i % self.checkpoint == 0 or i == max_iter - 1:
                obj_new = obj.item()
                self.vprint(f"\nInner iteration {i}")
                self.vprint(f'\th(W(model)): {h_val.item()}')
                self.vprint(f'\tscore(model): {obj_new}')
                if np.abs((obj_prev - obj_new) / obj_prev) <= tol:
                    pbar.update(max_iter - i)
                    break
                obj_prev = obj_new
            pbar.update(1)
        return True

    def fit(self,
            X: typing.Union[torch.Tensor, np.ndarray],
            lambda1: float = .02,
            lambda2: float = .005,
            T: int = 4,
            mu_init: float = .1,
            mu_factor: float = .1,
            s: float = 1.0,
            warm_iter: int = 5e4,
            max_iter: int = 8e4,
            lr: float = .0002,
            w_threshold: float = 0.3,
            checkpoint: int = 1000,
            ) -> np.ndarray:
        r"""
        Runs the DAGMA algorithm and fits the model to the dataset.

        Parameters
        ----------
        X : typing.Union[torch.Tensor, np.ndarray]
            :math:`(n,d)` dataset.
        lambda1 : float, optional
            Coefficient of the L1 penalty, by default .02.
        lambda2 : float, optional
            Coefficient of the L2 penalty, by default .005.
        T : int, optional
            Number of DAGMA iterations, by default 4.
        mu_init : float, optional
            Initial value of :math:`\mu`, by default 0.1.
        mu_factor : float, optional
            Decay factor for :math:`\mu`, by default .1.
        s : float, optional
            Controls the domain of M-matrices, by default 1.0.
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t < T`, by default 5e4.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.nonlinear.DagmaNonlinear.minimize` for :math:`t = T`, by default 8e4.
        lr : float, optional
            Learning rate, by default .0002.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold, by default 0.3.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations, by default 1000.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.


        .. important::

            If the output of :py:meth:`~dagma.nonlinear.DagmaNonlinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8)
            before raising an issue in github.
        """
        if type(X) == torch.Tensor:
            self.X = X.type(self.dtype)
        elif type(X) == np.ndarray:
            self.X = torch.from_numpy(X).type(self.dtype)
        else:
            ValueError("X should be numpy array or torch Tensor.")

        self.checkpoint = checkpoint
        mu = mu_init
        if type(s) == list:
            if len(s) < T:
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")
        with tqdm(total=(T - 1) * warm_iter + max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nDagma iter t={i + 1} -- mu: {mu}', 30 * '-')
                success, s_cur = False, s[i]
                inner_iter = int(max_iter) if i == T - 1 else int(warm_iter)
                model_copy = copy.deepcopy(self.model)
                lr_decay = False
                while success is False:
                    success = self.minimize(inner_iter, lr, lambda1, lambda2, mu, s_cur,
                                            lr_decay, pbar=pbar)
                    if success is False:
                        self.model.load_state_dict(model_copy.state_dict().copy())
                        lr *= 0.5
                        lr_decay = True
                        if lr < 1e-10:
                            break  # lr is too small
                        s_cur = 1
                mu *= mu_factor
        W_est = self.model.fc1_to_adj()
        # W_est[np.abs(W_est) < w_threshold] = 0
        return W_est

