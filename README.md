# 🧩 Self-Supervision is All You Need for Solving Rubik's Cube 

This repository contains code, models, and solutions as reported in the following paper:
> Takano, K. [Self-Supervision is All You Need for Solving Rubik's Cube](https://arxiv.org/abs/2106.03157). (2021) 

Briefly explained, the idea is to teach a neural network the stochastic process of scrambling the Rubik's Cube at random.
Despite its simplicity, the proposed method can yield optimal or near-optimal solutions.
Please read the paper for technical details.

**Demo is available on Replicate**: [replicate.com/kyo-takano/efficientcube](https://replicate.com/kyo-takano/efficientcube)

## Code
We provide ***standalone*** Jupyter Notebooks to run and test the proposed method on some goal-predefined combinatorial problems.
For full replications, please modify the hyperparameters.

<table>
    <thead>
        <tr>
            <th colspan=2>Section</th>
            <th>File</th>
            <th>Run</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1>Main</td>
            <td>Rubik's Cube</td>
            <td>
                <a href="./notebooks/main.ipynb">
                    <code>./notebooks/main.ipynb</code>
                    <a>
            </td>
            <td>
                <a href="https://colab.research.google.com/github/kyo-takano/EfficientCube/blob/main/notebooks/main.ipynb" rel="nofollow" target="_blank">
                    <img src="https://colab.research.google.com/assets/colab-badge.svg" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; min-height: 1rem;">
                </a>
                <a href="https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/EfficientCube/blob/main/notebooks/main.ipynb" rel="nofollow" target="_self">
                    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggle">
                </a>
                <!-- <a href="https://console.paperspace.com/github/kyo-takano/EfficientCube/blob/main/notebooks/main.ipynb">
                    <img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient" />
                </a> -->
            </td>
        </tr>
        <tr>
            <td rowspan=3>Appendix</td>
            <td white-space="nowrap">A: Assumption 1 on 2x2x2 Rubik's Cube</td>
            <td>
                <a href="./notebooks/appendix_a.ipynb">
                    <code>./notebooks/appendix_a.ipynb</code>
                    <a>
            </td>
            <td>
                <a href="https://colab.research.google.com/github/kyo-takano/EfficientCube/blob/main/notebooks/appendix_a.ipynb" rel="nofollow" target="_blank">
                    <img src="https://colab.research.google.com/assets/colab-badge.svg" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; min-height: 1rem;">
                </a>
                <a href="https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/EfficientCube/blob/main/notebooks/main.ipynb" rel="nofollow" target="_self">
                    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggle">
                </a>
                <!-- <a href="https://console.paperspace.com/github/kyo-takano/EfficientCube/blob/main/notebooks/appendix_a.ipynb">
                    <img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient" />
                </a> -->
            </td>
        </tr>
        <tr>
            <td>B: 15 Puzzle</td>
            <td>
                <a href="./notebooks/appendix_b.ipynb">
                    <code>./notebooks/appendix_b.ipynb</code>
                    <a>
            </td>
            <td>
                <a href="https://colab.research.google.com/github/kyo-takano/EfficientCube/blob/main/notebooks/appendix_b.ipynb" rel="nofollow" target="_blank">
                    <img src="https://colab.research.google.com/assets/colab-badge.svg" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; min-height: 1rem;">
                </a>
                <a href="https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/EfficientCube/blob/main/notebooks/appendix_b.ipynb" rel="nofollow" target="_self">
                    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggle">
                </a>
                <!-- <a href="https://console.paperspace.com/github/kyo-takano/EfficientCube/blob/main/notebooks/appendix_b.ipynb">
                    <img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient" />
                </a> -->
            </td>
        </tr>
        <tr>
            <td>C: 7x7 Lights Out</td>
            <td>
                <a href="./notebooks/appendix_c.ipynb" target="_blank">
                    <code>./notebooks/appendix_c.ipynb</code>
                    <a>
            </td>
            <td>
                <a href="https://colab.research.google.com/github/kyo-takano/EfficientCube/blob/main/notebooks/appendix_c.ipynb" rel="nofollow" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"
                        data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="max-width: 100%; min-height: 1rem;">
                </a>
                <a href="https://kaggle.com/kernels/welcome?src=https://github.com/kyo-takano/EfficientCube/blob/main/notebooks/appendix_c.ipynb" rel="nofollow" target="_self">
                    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Kaggle">
                </a>
                <!-- <a href="https://console.paperspace.com/github/kyo-takano/EfficientCube/blob/main/notebooks/appendix_c.ipynb">
                    <img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient" />
                </a> -->
            </td>
        </tr>
    </tbody>
</table>

## Models
We put [TorchScript](https://pytorch.org/docs/stable/jit.html) models as `./models/{cube3|puzzle15|lightsout7}/*steps_scripted.pth`.

Example usage (Rubik's Cube):
```python
# setup scramble and search parameters
scramble = """F U U L L B B F' U L L U R R D D L' B L L B' R R U U""" # Scramble for current *human* world record 
beam_width = 2**10
max_depth = 52 # arbitrary

# load the trained model
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(f"./models/cube3/1000000steps_scripted.pth").to(device)
model.eval()

# set up Rubik's Cube environment and apply the scramble
from utils import environments
env = environments.Cube3()
env.apply_scramble(scramble)

# execute a beam search
from utils import search
success, result = search.beam_search(env, model, max_depth, beam_width)

if success:
    print('Solution:', ' '.join(result['solutions']))
    print('Length:', len(result['solutions']))
    env.reset()
    env.apply_scramble(scramble)
    env.apply_scramble(result['solutions'])
    assert env.is_solved()
else:
    print('Failed')
```

## Solutions
Under `./results/`, problems have their own subdirectory containing Pickle file(s) (`./results/{cube3|puzzle15|lightsout7}/n={training_steps}.k={beam_width}.pkl`).

Each of the files holds a set of results (`solutions`, `times`, and `num_nodes_generated`) obtained with parameters `n` (number of training steps) and `k` (beam width). 
Please note that we only include the *actual* times taken per solution.

The DeepCubeA dataset is available from either [Code Ocean](http://doi.org/10.24433/CO.4958495.v1) or [GitHub](http://github.com/forestagostinelli/DeepCubeA/).

```python
import pickle
with open(filename, "rb") as f:
    data = pickle.load(f)
    solutions, times, num_nodes_generated = [data[k] for k in ['solutions', 'times', 'num_nodes_generated']]
    # solution_lengths = [len(e) for e in solutions]
```

### 📢 UPDATE on 15 Puzzle
> We additionally include updated results on 15 Puzzle, which are available as `./results/puzzle15/n=100000.k={beam_width}.pkl.update` (same filenames, followed by **`.update`**).
> 
> Two changes were made to the beam search implementation:
> 1. fixed a small error which allowed the selection of impossible moves (i.e., not affecting the problem state) at the very beginning of a search. This fix slightly improved search performance.
> 2. changed how to evaluate and sort candidates at every depth. \
> Instead of sorting moves by their probabilities (softmax-ed logits) at depth level, we tried sorting *paths* by their cumulative probabilities (`search.beam_search(..., __eval="cumprod")`).
> 
> These changes yielded significantly better performance than before, **solving all the test cases optimally with fewer nodes expanded than DeepCubeA**. When `beam_width=2**16`, an average of $2.54\times10^6$ nodes were expanded per optimal solution.

---

## Citing this work
If you use anything from this repository, please cite:
```bibtex
@article{takano2021self,
    author  = {Kyo Takano},
    title   = {Self-Supervision is All You Need for Solving Rubik's Cube},
    year    = {2021},
    journal = {arXiv},
    doi     = {10.48550/arXiv.2106.03157}
}
```
## Getting in Touch
Please contact the author at <code><a href="mailto:kyo.takano@mentalese.co" target="_blank">kyo.takano@mentalese.co</a></code> should you have any questions.
