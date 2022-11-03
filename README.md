# Self-Supervision is All You Need for Solving Rubik's Cube 

This repository contains code, models, and solutions as reported in the following paper:
> Takano, K. [Self-Supervision is All You Need for Solving Rubik's Cube](https://arxiv.org/abs/2106.03157). (2021) 

Briefly explained, the idea is to teach a neural network the stochastic process of scrambling the Rubik's Cube at random.
Despite its simplicity, the proposed method can yield optimal or near-optimal solutions.
Please read the paper for technical details.

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

Example_usage (Rubik's Cube):
```python
# setup scramble and search parameters
scramble = """F U U L L B B F' U L L U R R D D L' B L L B' R R U U""" # Scramble for current *human* world record 
beam_width = 2**8
max_depth = 50
__eval = 'logits' # you may want to change this to 'cumprod' to get a better solution, especially for 15 puzzle.

# load the trained model
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(f"./models/cube3/1000000steps_scripted.pth")
model.eval()

# load Rubik's Cube environment and apply the scramble
from utils import environments
env = environments.Cube3()
env.apply_scramble(scramble)

from utils import search
success, result = search.beam_search(env, model, max_depth, beam_width, __eval==__eval)
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
Under `./results/`, problems have their own subdirectory containing Pickle file(s) (`./results/{cube3|puzzle15|lightsout7}/*.pkl`).
Each of the files holds a set of results (`solutions`, `times`, and `num_nodes_generated`) obtained with parameters `n` (number of training steps) and `k` (beam width).
```python
import pickle
with open(filename, "rb") as f:
    data = pickle.load(f)
    solutions, times, num_nodes_generated = [data[k] for k in ['solutions', 'times', 'num_nodes_generated']]
    # solution_lengths = [len(e) for e in solutions]
```
Please note that we only include the *actual* times taken per solution.

The DeepCubeA dataset is available from either [Code Ocean](http://doi.org/10.24433/CO.4958495.v1) or [GitHub](http://github.com/forestagostinelli/DeepCubeA/).

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
