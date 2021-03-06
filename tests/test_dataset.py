import numpy as np
import pytest
import torch
import os

from collections import deque
from sklearn.model_selection import train_test_split
from d3rlpy.dataset import MDPDataset, Episode, Transition, TransitionMiniBatch
from d3rlpy.dataset import _stack_frames


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_frames', [4])
@pytest.mark.parametrize('as_tensor', [False, True])
def test_stack_frames(data_size, observation_shape, action_size, n_frames,
                      as_tensor):
    if as_tensor:
        observations = torch.rand(data_size, *observation_shape)
    else:
        observations = np.random.random((data_size, *observation_shape))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random((data_size, 1))

    episode = Episode(observation_shape=observation_shape,
                      action_size=action_size,
                      observations=observations,
                      actions=actions,
                      rewards=rewards)

    image_size = observation_shape[1:]
    n_channels = n_frames * observation_shape[0]
    stacked_shape = (n_channels, *image_size)

    if as_tensor:
        padding = torch.zeros(n_frames - 1, *observation_shape)
        padded_observations = torch.cat([padding, observations])
    else:
        padding = np.zeros((n_frames - 1, *observation_shape))
        padded_observations = np.vstack([padding, observations])

    for i, transition in enumerate(episode.transitions):
        observation, next_observation = _stack_frames(transition, n_frames)

        # create reference stacked observation
        head_index = i
        tail_index = head_index + n_frames
        window = padded_observations[head_index:tail_index]
        next_window = padded_observations[head_index + 1:tail_index + 1]

        if as_tensor:
            ref_observation = window.view(stacked_shape)
            ref_next_observation = next_window.view(stacked_shape)
            assert observation.shape == ref_observation.shape
            assert next_observation.shape == ref_next_observation.shape
            assert torch.all(observation == ref_observation)
            assert torch.all(next_observation == ref_next_observation)
        else:
            ref_observation = np.vstack(window)
            ref_next_observation = np.vstack(next_window)
            assert observation.shape == ref_observation.shape
            assert next_observation.shape == ref_next_observation.shape
            assert np.all(observation == ref_observation)
            assert np.all(next_observation == ref_next_observation)


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_episodes', [4])
@pytest.mark.parametrize('discrete_action', [True, False])
def test_mdp_dataset(data_size, observation_size, action_size, n_episodes,
                     discrete_action):
    observations = np.random.random((data_size, observation_size))
    rewards = np.random.uniform(-10.0, 10.0, size=data_size)
    n_steps = data_size // n_episodes
    terminals = np.array(([0] * (n_steps - 1) + [1]) * n_episodes)

    if discrete_action:
        actions = np.random.randint(action_size, size=data_size)
        ref_action_size = np.max(actions) + 1
    else:
        actions = np.random.random((data_size, action_size))
        ref_action_size = action_size

    dataset = MDPDataset(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         terminals=terminals,
                         discrete_action=discrete_action)

    # check MDPDataset methods
    assert np.all(dataset.observations == observations)
    assert np.all(dataset.actions == actions)
    assert np.all(dataset.rewards == rewards)
    assert np.all(dataset.terminals == terminals)
    assert dataset.size() == n_episodes
    assert dataset.get_action_size() == action_size
    assert dataset.get_observation_shape() == (observation_size, )
    assert dataset.is_action_discrete() == discrete_action

    # check stats
    ref_returns = []
    for i in range(n_episodes):
        episode_return = 0.0
        for j in range(1, n_steps):
            episode_return += rewards[j + i * n_steps]
        ref_returns.append(episode_return)

    stats = dataset.compute_stats()
    return_stats = stats['return']
    assert np.allclose(return_stats['mean'], np.mean(ref_returns))
    assert np.allclose(return_stats['std'], np.std(ref_returns))
    assert np.allclose(return_stats['min'], np.min(ref_returns))
    assert np.allclose(return_stats['max'], np.max(ref_returns))
    reward_stats = stats['reward']
    assert np.allclose(reward_stats['mean'], np.mean(rewards))
    assert np.allclose(reward_stats['std'], np.std(rewards))
    assert np.allclose(reward_stats['min'], np.min(rewards))
    assert np.allclose(reward_stats['max'], np.max(rewards))
    observation_stats = stats['observation']
    assert np.all(observation_stats['mean'] == np.mean(observations, axis=0))
    assert np.all(observation_stats['std'] == np.std(observations, axis=0))
    if discrete_action:
        freqs, action_ids = stats['action']['histogram']
        assert np.sum(freqs) == data_size
        assert list(action_ids) == [i for i in range(action_size)]
    else:
        action_stats = stats['action']
        assert np.all(action_stats['mean'] == np.mean(actions, axis=0))
        assert np.all(action_stats['std'] == np.std(actions, axis=0))
        assert np.all(action_stats['min'] == np.min(actions, axis=0))
        assert np.all(action_stats['max'] == np.max(actions, axis=0))
        assert len(action_stats['histogram']) == action_size
        for freqs, _ in action_stats['histogram']:
            assert np.sum(freqs) == data_size

    # check episodes exported from dataset
    episodes = dataset.episodes
    assert len(episodes) == n_episodes
    for i, e in enumerate(dataset.episodes):
        assert isinstance(e, Episode)
        assert e.size() == n_steps - 1
        head = i * n_steps
        tail = head + n_steps
        assert np.all(e.observations == observations[head:tail])
        assert np.all(e.actions == actions[head:tail])
        assert np.all(e.rewards == rewards[head:tail])
        assert e.get_observation_shape() == (observation_size, )
        assert e.get_action_size() == ref_action_size

    # check list-like behaviors
    assert len(dataset) == n_episodes
    assert dataset[0] is dataset.episodes[0]
    for i, episode in enumerate(dataset.episodes):
        assert isinstance(episode, Episode)
        assert episode is dataset.episodes[i]

    # check append
    dataset.append(observations, actions, rewards, terminals)
    assert len(dataset) == 2 * n_episodes
    assert dataset.observations.shape == (2 * data_size, observation_size)
    assert dataset.rewards.shape == (2 * data_size, )
    assert dataset.terminals.shape == (2 * data_size, )
    if discrete_action:
        assert dataset.actions.shape == (2 * data_size, )
    else:
        assert dataset.actions.shape == (2 * data_size, action_size)

    # check extend
    another_dataset = MDPDataset(observations, actions, rewards, terminals,
                                 discrete_action)
    dataset.extend(another_dataset)
    assert len(dataset) == 3 * n_episodes
    assert dataset.observations.shape == (3 * data_size, observation_size)
    assert dataset.rewards.shape == (3 * data_size, )
    assert dataset.terminals.shape == (3 * data_size, )
    if discrete_action:
        assert dataset.actions.shape == (3 * data_size, )
    else:
        assert dataset.actions.shape == (3 * data_size, action_size)

    # check clip_reward
    dataset.clip_reward(-1.0, 1.0)
    assert rewards[rewards > 1.0].sum() != 0
    assert rewards[rewards < -1.0].sum() != 0
    assert dataset.rewards[dataset.rewards > 1.0].sum() == 0
    assert dataset.rewards[dataset.rewards < -1.0].sum() == 0

    # check dump and load
    dataset.dump(os.path.join('test_data', 'dataset.h5'))
    new_dataset = MDPDataset.load(os.path.join('test_data', 'dataset.h5'))
    assert np.all(dataset.observations == new_dataset.observations)
    assert np.all(dataset.actions == new_dataset.actions)
    assert np.all(dataset.rewards == new_dataset.rewards)
    assert np.all(dataset.terminals == new_dataset.terminals)
    assert dataset.discrete_action == new_dataset.discrete_action
    assert len(dataset) == len(new_dataset)

    # check as_tensor
    dataset = MDPDataset(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         terminals=terminals,
                         discrete_action=discrete_action,
                         as_tensor=True)
    assert isinstance(dataset.observations, torch.Tensor)
    for episode in dataset:
        assert isinstance(episode.observations, torch.Tensor)
        for transition in episode:
            assert isinstance(transition.observation, torch.Tensor)
            assert isinstance(transition.next_observation, torch.Tensor)


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
def test_episode(data_size, observation_size, action_size):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random((data_size, 1))

    episode = Episode(observation_shape=(observation_size, ),
                      action_size=action_size,
                      observations=observations,
                      actions=actions,
                      rewards=rewards)

    # check Episode methods
    assert np.all(episode.observations == observations)
    assert np.all(episode.actions == actions)
    assert np.all(episode.rewards == rewards)
    assert episode.size() == data_size - 1
    assert episode.get_observation_shape() == (observation_size, )
    assert episode.get_action_size() == action_size
    assert episode.compute_return() == np.sum(rewards[1:])

    # check transitions exported from episode
    assert len(episode.transitions) == data_size - 1
    for i, t in enumerate(episode.transitions):
        assert isinstance(t, Transition)
        assert t.observation_shape == (observation_size, )
        assert t.action_size == action_size
        assert np.all(t.observation == observations[i])
        assert np.all(t.action == actions[i])
        assert t.reward == rewards[i]
        assert np.all(t.next_observation == observations[i + 1])
        assert np.all(t.next_action == actions[i + 1])
        assert t.next_reward == rewards[i + 1]
        assert t.terminal == (1.0 if (i == data_size - 2) else 0.0)

    # check forward pointers
    count = 1
    transition = episode[0]
    while transition.next_transition:
        transition = transition.next_transition
        count += 1
    assert count == data_size - 1

    # check backward pointers
    count = 1
    transition = episode[-1]
    while transition.prev_transition:
        transition = transition.prev_transition
        count += 1
    assert count == data_size - 1

    # check list-like bahaviors
    assert len(episode) == data_size - 1
    assert episode[0] is episode.transitions[0]
    for i, transition in enumerate(episode):
        assert isinstance(transition, Transition)
        assert transition is episode.transitions[i]


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('as_tensor', [True, False])
@pytest.mark.parametrize('n_frames', [1, 4])
def test_transition_minibatch(data_size, observation_shape, action_size,
                              as_tensor, n_frames):
    if as_tensor:
        observations = torch.rand(data_size, *observation_shape)
    else:
        observations = np.random.random((data_size, *observation_shape))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random((data_size, 1))

    episode = Episode(observation_shape=observation_shape,
                      action_size=action_size,
                      observations=observations,
                      actions=actions,
                      rewards=rewards)

    if len(observation_shape) == 3:
        n_channels = n_frames * observation_shape[0]
        image_size = observation_shape[1:]
        batched_observation_shape = (data_size - 1, n_channels, *image_size)
    else:
        batched_observation_shape = (data_size - 1, *observation_shape)

    batch = TransitionMiniBatch(episode.transitions, n_frames)
    assert batch.observations.shape == batched_observation_shape
    assert batch.next_observations.shape == batched_observation_shape
    for i, t in enumerate(episode.transitions):
        observation = batch.observations[i]
        next_observation = batch.next_observations[i]
        if as_tensor:
            observation = observation.numpy()
            next_observation = next_observation.numpy()

        if n_frames == 1:
            assert np.allclose(observation, t.observation)
            assert np.allclose(next_observation, t.next_observation)

        assert np.all(batch.actions[i] == t.action)
        assert np.all(batch.rewards[i] == t.reward)
        assert np.all(batch.next_actions[i] == t.next_action)
        assert np.all(batch.next_rewards[i] == t.next_reward)
        assert np.all(batch.terminals[i] == t.terminal)

    # check list-like behavior
    assert len(batch) == data_size - 1
    assert batch[0] is episode.transitions[0]
    for i, transition in enumerate(batch):
        assert isinstance(transition, Transition)
        assert transition is episode.transitions[i]


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_episodes', [10])
@pytest.mark.parametrize('test_size', [0.2])
def test_dataset_with_sklearn(data_size, observation_size, action_size,
                              n_episodes, test_size):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random(data_size)
    n_steps = data_size // n_episodes
    terminals = np.array(([0] * (n_steps - 1) + [1]) * n_episodes)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # check compatibility with train_test_split
    train_episodes, test_episodes = train_test_split(dataset,
                                                     test_size=test_size)
    assert len(train_episodes) == int(n_episodes * (1.0 - test_size))
    assert len(test_episodes) == int(n_episodes * test_size)
