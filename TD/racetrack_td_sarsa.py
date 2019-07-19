import os
import math
from collections import defaultdict

import numpy as np

from racetrack_env import RacetrackEnv, Map, REWARD_SUCCESS

MAX_EPISODE = 10000  # Recommend: 10000 for E-Greedy, 300000 for UCB
MAX_STEP = 70
EGREEDY_EPS = 0.1
UCB = False  # True: UCB, False: E-Greedy
UCB_C = 10.0
GAMMA = 0.99
ALPHA = 0.1
SHOW_TERM = 10001
SPOLICY = "UCB" if UCB else "EGreedy"
SAVE_FILENM = "Racetrack_sarsa{}.sav".format(SPOLICY)


def make_env():
    with open('racetrack_map_4.txt', 'r') as f:
        amap = Map(f.read())

    vel_info = (
        0, 3,  # vx min / max
        -3, 3   # vy min / max
    )

    env = RacetrackEnv(amap, vel_info, MAX_STEP)
    return env


def egreedy_policy(env, Q, state, e_no, test_action=None):
    aprobs = Q[state]
    if test_action is not None:
        action = test_action
    else:
        action = np.random.choice(np.flatnonzero(aprobs == aprobs.max()))
    nA = env.action_space.n
    eps = EGREEDY_EPS * (1 - float(e_no) / MAX_EPISODE)
    # eps = EGREEDY_EPS
    A = np.ones(nA) * eps / nA
    A[action] += (1.0 - eps)
    return A


def egreedy_action(aprobs, nA):
    return np.random.choice(range(nA), p=aprobs)


def test_egreedy_policy():
    env = make_env()
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    best_action = 1
    state = None
    aprobs = egreedy_policy(env, Q, state, 1, best_action)
    assert np.array_equal(aprobs, np.array([0.02, 0.92, 0.02, 0.02, 0.02]))
    n = 0
    acnt = defaultdict(int)
    TRY_CNT = 100
    while n < TRY_CNT:
        action = np.random.choice(range(nA), p=aprobs)
        acnt[action] += 1
        n += 1
    EPS_CNT = 100 * EGREEDY_EPS
    assert TRY_CNT - acnt[best_action] < 2 * EPS_CNT


def ucb_policy(env, Q, N, state, t, e, test_action=None):
    rv = Q[state] + UCB_C * np.sqrt(math.log(t) / N[state])
    return rv


def ucb_action(aprobs, state, N, update=True):
    action = np.random.choice(np.flatnonzero(aprobs == aprobs.max()))
    if update:
        N[state][action] += 1
    return action


def test_ucb_policy():
    env = make_env()
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.ones(nA))
    best_action = 1
    state = None
    t = 1

    acnt = defaultdict(int)
    for i in range(nA):
        aprobs = ucb_policy(env, Q, N, state, t, 1, best_action)
        action = ucb_action(aprobs, state, N)
        acnt[action] += 1
        t += 1

    assert list(acnt.values()) == [1] * nA


def make_greedy_policy(Q):
    def func(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return func


def _run_step(env, Q, N, state, action, nA, n_episode, n_step):

    nstate, reward, done, _ = env.step(action)

    if UCB:
        naprobs = ucb_policy(env, Q, N, nstate, n_step + 2, n_episode + 1)
        naction = ucb_action(naprobs, nstate, N, False)
    else:
        naprobs = egreedy_policy(env, Q, nstate, n_episode + 1)
        naction = np.random.choice(range(nA), p=naprobs)

    v = Q[state][action]
    nv = Q[nstate][naction]
    td_target = reward + GAMMA * nv
    td_delta = td_target - v
    Q[state][action] += ALPHA * td_delta
    return nstate, naction, reward, done  # naction instead of action by Sarsa algorithm


def _print_policy_progress(Q, state, N, action):
    if UCB:
        print("  ", state, Q[state], N[state], action)
    else:
        print("  ", state, Q[state], action)


def _print_done_msg(reward):
    if reward == REWARD_SUCCESS:
        print("   SUCCESS!!")
    else:
        print("   DONE")


def learn_Q(env):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.ones(nA))
    rewards_list = []

    for n_episode in range(MAX_EPISODE):
        state = env.reset()
        if UCB:
            action = action = np.random.choice(range(nA))
        else:
            naprobs = egreedy_policy(env, Q, state, n_episode + 1)
            action = np.random.choice(range(nA), p=naprobs)

        show = (n_episode + 1) % SHOW_TERM == 0
        if show:
            print("========== Policy: {}, Episode: {} / {} ==========".
                  format(SPOLICY, n_episode + 1, MAX_EPISODE))

        for n_step in range(MAX_STEP):
            state, action, reward, done = _run_step(env, Q, N, state, action, nA,
                                                    n_episode, n_step)
            if reward == REWARD_SUCCESS:
                rewards_list.append(1.)
            if show:
                _print_policy_progress(Q, state, N, action)
            if done:
                if show:
                    _print_done_msg(reward)
                break
    print("The average of rewards: {}".format(np.sum(rewards_list)/MAX_EPISODE))
    #print(np.sum(rewards_list), len(rewards_list))
    return Q


def run():
    env = make_env()
    Q = None

    if os.path.isfile(SAVE_FILENM):
        ans = input("Saved file '{}' exists. Load the file and play? (Y/N): "
                    .format(SAVE_FILENM))
        if ans.lower().startswith('y'):
            Q = env.load(SAVE_FILENM)
        else:
            print("Start new learning!")

    if Q is None:
        Q = learn_Q(env)
        env.save(Q, SAVE_FILENM)

    play_policy = make_greedy_policy(Q)
    env.play(play_policy, 1)


if __name__ == "__main__":
    run()
