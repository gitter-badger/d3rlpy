import numpy as np
import argparse
import time

from d3rlpy.dataset import MDPDataset, TransitionMiniBatch
from d3rlpy.algos import DQN

DATA_SIZE = 10000
ACTION_SIZE = 4
OBSERVATION_SHAPE = (DATA_SIZE, 1, 84, 84)
STACKED_SHAPE = (4, 84, 84)
BATCH_SIZE = 100

def make_batch(transitions, batch_size, n_frames):
    indices = np.random.permutation(np.arange(len(transitions)))
    n_iterations = len(transitions) // batch_size
    for i in range(n_iterations):
        head_index = i * batch_size
        tail_index = head_index + batch_size
        batch = []
        for index in indices[head_index:tail_index]:
            batch.append(transitions[index])
        yield TransitionMiniBatch(batch, n_frames=n_frames)


def main(args):
    # prepare dataset
    observations = np.random.randint(256, size=OBSERVATION_SHAPE, dtype=np.uint8)
    actions = np.random.randint(ACTION_SIZE, size=DATA_SIZE)
    rewards = np.random.random((DATA_SIZE, 1))
    terminals = (np.arange(DATA_SIZE) % 100) == 0

    dataset = MDPDataset(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         terminals=terminals,
                         discrete_action=True,
                         as_tensor=args.use_gpu_for_dataset,
                         device=args.gpu)

    # prepare algorithm
    dqn = DQN(n_frames=4, batch_size=BATCH_SIZE, use_gpu=args.gpu)
    dqn.create_impl(STACKED_SHAPE, ACTION_SIZE)

    # prepare transitions
    transitions = []
    for episode in dataset:
        transitions += episode.transitions

    # batch loop
    start = time.time()
    for i, batch in enumerate(make_batch(transitions, BATCH_SIZE, 4)):
        dqn.update(0, i, batch)
    print('elapsed time: ', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--use-gpu-for-dataset', action='store_true')
    args = parser.parse_args()
    main(args)
