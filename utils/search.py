
import time
import torch
import numpy as np
from scipy.special import softmax
from copy import deepcopy
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def greedy_search(env, model, max_depth, skip_redundant_moves=True):
    return beam_search(env, model, max_depth, 1, __eval = 'logits', skip_redundant_moves=True)

def beam_search(
        env,
        model,
        max_depth,
        beam_width,
        __eval = 'logits',
        skip_redundant_moves=True,
    ):
    """
    Best-first search algorithm.
    Input:
        env: A scrambled instance of the given environment. 
        beam_width: Number of top solutions to return per depth.
        max_depth: Maximum depth of the search tree.
        __eval: Evaluation method for sorting nodes to expand, based on DNN outputs: 'softmax', 'logits', or 'cumprod'. 
        skip_redundant_moves: If True, skip redundant moves.
        ...
    Output: 
        if solved successfully:
            True, {'solutions':solution path, "num_nodes":number of nodes expanded, "times":time taken to solve}
        else:
            False, None
    """
    env_class_name = env.__class__.__name__
    assert env_class_name in ['Cube3','Puzzle15','LightsOut7']
    with torch.no_grad():
        # metrics
        num_nodes, time_0 = 0, time.time()
        candidates = [
            {"state":deepcopy(env.state), "path":[], "value":1.}
        ] # list of dictionaries

        for depth in range(max_depth+1):
            # TWO things at a time for every candidate: 1. check if solved & 2. add to batch_x
            batch_x = np.zeros((len(candidates), env.state.shape[-1]), dtype=np.int64)
            for i,c in enumerate(candidates):
                c_path, env.state = c["path"], c["state"]
                if c_path:
                    env.finger(c_path[-1])
                    num_nodes += 1
                    if env.is_solved():
                        return True, {'solutions':c_path, "num_nodes":num_nodes, "times":time.time()-time_0}
                batch_x[i, :] = env.state

            # after checking the nodes expanded at the deepest    
            if depth==max_depth:
                print("Solution not found.")
                return False, None

            # make predictions with the trained DNN
            if len(candidates)<2**17:
                batch_x = torch.from_numpy(batch_x).to(device)
                batch_p = model(batch_x).to("cpu").detach().numpy()
            else:
                # split the batch so as to avoid 'CUDA out of memory' error.
                batch_p = [
                    model(torch.from_numpy(batch_x_mini).to(device)).to('cpu').detach().numpy() 
                    for batch_x_mini in np.split(batch_x, len(candidates)//(2**16))
                ]
                batch_p = np.concatenate(batch_p)
            if __eval in ["softmax","cumprod"]:
                batch_p = softmax(batch_p, axis=1)

            # loop over candidates
            candidates_next_depth = []  # storage for the depth-level candidates storing (path, value, index).
            for i, c in enumerate(candidates):
                c_path = c["path"]
                value_distribution = batch_p[i, :] # output logits for the given state
                if __eval=="cumprod":
                    value_distribution *= c["value"] # multiply the cumulative probability so far of the expanded path

                for m, value in zip(env.moves_inference, value_distribution): # iterate over all possible moves.
                    # predicted value to expand the path with the given move.

                    if env_class_name=='Cube3':
                        if c_path and skip_redundant_moves:
                            if env.metric=='QTM':
                                if m not in env.moves_available_after[c_path[-1]]:
                                    # Two mutually canceling moves
                                    continue
                                elif len(c_path) > 1:
                                    if c_path[-2] == c_path[-1] == m:
                                        # three subsequent same moves
                                        continue
                                    elif (
                                        c_path[-2][0] == m[0]
                                        and len(c_path[-2] + m) == 3
                                        and c_path[-1][0] == env.pairing[m[0]]
                                    ):
                                        # Two mutually canceling moves sandwiching an opposite face move
                                        continue
                            elif env.metric=='HTM':
                                if c_path:
                                    if skip_redundant_moves:
                                        if m[0] == c_path[-1][0]:
                                            # Two mutually canceling moves
                                            continue
                                        elif len(c_path)>1:
                                            if c_path[-2][0] == m[0] and c_path[-1][0] == env.pairing[m[0]]:
                                                # Two mutually canceling moves sandwiching an opposite face move
                                                continue
                            else:
                                raise
                    elif env_class_name=='Puzzle15':
                        # remove (physically) illegal moves, whether you like it or not
                        target_loc = np.where(c['state'].reshape(4, 4) == 0)
                        if m=="R":
                            if not target_loc[1]: # zero_index (empty slot) on the left
                                continue
                        elif m=="D":
                            if not target_loc[0]: # on the top
                                continue
                        elif m=="U":
                            if target_loc[0]==3: # on the bottom
                                continue
                        elif m=="L":
                            if target_loc[1]==3: # on the right
                                continue
                        if c_path:
                            if skip_redundant_moves:
                                # Two cancelling moves
                                if env.pairing[c_path[-1]] == m:
                                    continue
                    elif env_class_name=='LightsOut7':
                        if skip_redundant_moves:
                            # logically meaningless operation
                            if m in c_path:
                                continue
                    else:
                        raise

                    # add to the next-depth candidates unless 'continue'd.
                    candidates_next_depth.append({
                        'state':deepcopy(c['state']),
                        "path": c_path+[m],
                        "value":value,
                    })

            # sort potential paths by expected values and renew as 'candidates'
            candidates = sorted(candidates_next_depth, key=lambda item: -item['value'])
            # if the number of candidates exceed that of beam width 'beam_width'
            candidates = candidates[:beam_width]
