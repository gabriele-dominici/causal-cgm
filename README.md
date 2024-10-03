# Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning

Repository related to the [**"Causal Concept Graph Models: Beyond Causal Opacity in Deep Learning"**](https://arxiv.org/abs/2405.16507) paper.

### Instruction to set up the repository

**Requirement**: Python 3.9+

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Datasets

To reproduce the experiments discussed in out paper, you will have to download the CelebA dataset found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and the entire directory should be placed into ```./datasets/celeba/```.

To run all the experiments, you should first run the following: 
(you don't need to run the first one as we already provided the files inside ```./embeddings/dsprites/```)

```
python3 experiments/dsprites/dataset.py
python3 experiments/celeba/dataset.py
```

### Running experiments
```
python3 experiments/checkmark/run.py
python3 experiments/dsprites/run.py
python3 experiments/celeba/run.py
```


