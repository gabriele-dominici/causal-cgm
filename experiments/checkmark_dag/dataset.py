import torch


def toy_problem(n_samples=10, seed=42):
    torch.manual_seed(seed)
    A = torch.randint(0, 2, (n_samples,), dtype=torch.bool)
    torch.manual_seed(seed + 1)
    B = torch.randint(0, 2, (n_samples,), dtype=torch.bool)

    # Column C is true if B is true, randomly true/false if B is false
    C = ~B

    # Column D is true if A or C is true, randomly true/false if both are false
    D = A & C

    # Combine all columns into a matrix
    return torch.stack((A, B, C, D), dim=1).float()


def checkmark_dataset(n_samples=10, seed=42, perturb=0.1, return_y=False):
    x = toy_problem(n_samples, seed)
    c = x.clone()
    torch.manual_seed(seed)
    x = x * 2 - 1 + torch.randn_like(x) * perturb

    dag = torch.FloatTensor([[0, 0, 0, 1],  # A influences D
                            [0, 0, 1, 0],  # B influences C
                            [0, 0, 0, 1],  # C influences D
                            [0, 0, 0, 0],  # D doesn't influence others
                            ])

    if return_y:
        return x, c[:, [0, 1, 2]], dag, c[:, 3]
    else:
        return x, c, dag


