# -*- coding: utf8 -*-
import os
import time
from collections import defaultdict
import sys

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


N_PLAYRUN = 3
REWARD_DEFAULT = 0
REWARD_SUCCESS = 10
REWARD_OUT = -10

def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'OS X',
        'win32' : 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]


class Map(object):
    def __init__(self, data, road_t='*', road_trep='.', block_t=' ',
                 block_trep='#', h_mgn=1, v_mgn=1, start_t='s', finish_t='f'):
        """Initialize map object

        Args:
            data(str): Map data
            road_t: Road tile where car can pass
            road_trep: Replace tile of road for rendering
            block_t: Block tile where car can not pass
            block_trep: Replace tile of block for rendering
            h_mgn: Horizontal margin size (filled with block tile)
            v_mgn: Horizontal margin size (filled with block tile)

        """
        self.road_t = road_t
        self.road_trep = road_trep
        self.block_t = block_t
        self.block_trep = block_trep
        self.h_mgn, self.v_mgn = h_mgn, v_mgn
        self.start_t, self.finish_t = start_t, finish_t
        self.start_pts = []
        self.finish_pts = []
        self.road_pts = []
        self.car_t = 'O'
        self.loads(data)

    def store_attrs(self, height, line):
        y = height
        x = 1
        for c in line:
            if c == self.road_t:
                self.road_pts.append((x, y))
            elif c == self.start_t:
                self.start_pts.append((x, y))
                # self.road_pts.append((x, y))
            elif c == self.finish_t:
                self.finish_pts.append((x, y))
                # self.road_pts.append((x, y))
            x += 1

    def loads(self, data):
        width = 0
        height = 0
        rdata = []

        # analyze map data & make track(road) data
        for line in data.splitlines():
            lcnt = len(line.strip())
            if lcnt == 0:
                continue
            width = max(width, len(line))
            height += 1
            self.store_attrs(height, line)
        self.width, self.height = width, height

        def h_marginate(line):
            mgn = self.block_trep * self.h_mgn
            return "{}{}{}".format(mgn, line, mgn)

        def v_marginate():
            for i in range(self.v_mgn):
                mdata = h_marginate(self.block_trep * self.width)
                rdata.append(mdata)

        def _make_rdata_line(line):
            line = line.ljust(self.width)
            line = line.replace(self.road_t, self.road_trep)
            line = line.replace(self.block_t, self.block_trep)
            return line

        # top vertical margin
        v_marginate()

        # make render map data
        for line in data.splitlines():
            cnt = len(line.strip())
            if cnt == 0:
                continue
            rline = h_marginate(_make_rdata_line(line))
            rdata.append(rline)

        # bottom vertical margin
        v_marginate()

        self.rdata = rdata

    def make_draw_data(self, car_x, car_y):
        assert car_x > 0 and car_x <= self.width
        assert car_y > 0 and car_y <= self.height

        rdata = self.rdata[:]
        cx = car_x + self.h_mgn - 1
        cy = car_y + self.v_mgn - 1
        line = rdata[cy]
        rdata[cy] = line[:cx] + self.car_t + line[cx+1:]
        return '\n'.join(rdata)

    def in_road(self, x, y):
        assert x > 0 and x <= self.width
        assert y > 0 and y <= self.height
        return (x, y) in self.road_pts


def test_map():
    data = """
      *******f
   **********f
 ************f
 ************f
*************f
*********
********
********
********
********
ssssssss
    """
    m = Map(data, h_mgn=2, v_mgn=2)
    assert m.width == 14
    assert m.height == 11
    assert m.rdata[0] == '##################'
    assert m.rdata[2] == '########.......f##'
    assert m.rdata[-4] == '##........########'
    assert m.rdata[-3] == '##ssssssss########'
    assert m.rdata[-1] == '##################'

    assert m.start_pts == [(1, 11), (2, 11), (3, 11), (4, 11), (5, 11),
                           (6, 11), (7, 11), (8, 11)]
    assert m.finish_pts == [(14, 1), (14, 2), (14, 3), (14, 4), (14, 5)]
    assert m.make_draw_data(4, 11).splitlines()[-3] == '##sssOssss########'
    assert m.in_road(1, 11)
    assert m.in_road(14, 2)
    assert not m.in_road(1, 1)


log_level = 1


def debug(msg):
    if log_level > 2:
        print(msg)


def info(msg):
    if log_level > 1:
        print(msg)


class RacetrackEnv(gym.Env):
    """Simple racetrack environment

    Note:
        You want to to as fast as possible, but not to run off the track.
        Velocity & position of the car is descrete.
        Actions are to set velocity as +1, 0, -1 for X, Y
        Nine actions for a episode.
        Rewards are 0 for each step, 1 when finished.
    """
    def __init__(self, amap, vel_info, max_step):

        self.amap = amap
        self.vx_min, self.vx_max, self.vy_min, self.vy_max = vel_info
        self.max_step = max_step

        # Actions
        #    No velocity (0)
        #    X velocity +1, -1 (1, 2)
        #    Y velocity +1, -1 (3, 4)
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Tuple((
            spaces.Discrete(amap.width),  # px
            spaces.Discrete(amap.height),  # py
            spaces.Discrete(self.count_vx()),  # vx (0~4)
            spaces.Discrete(self.count_vy())))  # vy (0~4)
        self._seed()

        # Start the first game
        self._reset()

        self.E = {(px, py):0 for px in range(1, amap.width+1) for py in range(1, amap.height+1)}

    def count_vx(self):
        return self.vx_max - self.vx_min + 1

    def count_vy(self):
        return self.vy_max - self.vy_min + 1

    def total_states(self):
        return 4

    def regulate_probs(self, observation, A):
        _, _, vx, vy = observation
        prob = np.copy(A)

        if vx <= self.vx_min:
            prob[2] = 0
        elif vx >= self.vx_max:
            prob[1] = 0

        if vy <= self.vy_min:
            prob[4] = 0
        elif vy >= self.vy_max:
            prob[3] = 0

        if vx == 0 and vy == 0:
            prob[0] = 0

        # print(prob / prob.sum())
        return prob / prob.sum()

    def get_start(self):
        scnt = len(self.amap.start_pts)
        sidx = self.np_random.randint(scnt)
        return self.amap.start_pts[sidx]

    def get_exploring_start(self):
        scnt = len(self.amap.road_pts)
        sidx = self.np_random.randint(scnt)
        vx = np.random.randint(self.vx_min, self.vx_max)
        vy = np.random.randint(self.vy_min, self.vy_max)
        return self.amap.road_pts[sidx], vx, vy

    def _seed(self, seed=None):
        # random seed에 맞는 random 객체 얻어둠
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        debug("step action {}".format(action))
        assert self.action_space.contains(action)

        # X
        if action == 1:
            self.vx += 1
        elif action == 2:
            self.vx -= 1

        # Y
        if action == 3:
            self.vy += 1
        elif action == 4:
            self.vy -= 1

        prev_px, prev_py = self.px, self.py

        # print(self.vx, self.vy)
        self.vx = max(min(self.vx, self.vx_max), self.vx_min)
        self.vy = max(min(self.vy, self.vy_max), self.vy_min)
        # print("->", self.vx, self.vy)
        self.px += self.vx
        self.py += self.vy
        self.px = max(1, min(self.amap.width, self.px))
        self.py = max(1, min(self.amap.height, self.py))

        self.n_action += 1
        reward = REWARD_DEFAULT
        self.ppx = self.px
        done = False

        if (self.px, self.py) in self.amap.finish_pts:
            info("  Finished: {}".format(self._get_obs()))
            reward = REWARD_SUCCESS
            done = True
        elif (self.px, self.py) not in self.amap.road_pts:
            debug("  out of track with [{}]. restart!".format(self._get_obs()))
            if self.n_action != self.max_step:
                done = True
                reward = REWARD_OUT

        elif self.px == prev_px and self.py == prev_py:
            reward = -1

        if self.n_action == self.max_step:
            debug("  Out of turn")
            done = True
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (self.px, self.py, self.vx, self.vy)

    def race_start(self):
        self.px, self.py = self.get_start()
        self.ppx = self.px
        self.vx = self.vy = 0

    def _reset(self):
        self.race_start()
        self.n_action = 0
        # print("_reset")
        return self._get_obs()

    def score(self, policy):
        return self.play(policy, 0)

    def play(self, policy, render_type=2):
        n_succ = 0
        if render_type == 2:
            from IPython.display import clear_output

        for i in range(N_PLAYRUN):
            state = self.reset()
            done = False
            step = 1
            while not done:
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                state, reward, done, _ = self.step(action)
                if render_type > 0:
                    sys_os = get_platform()
                    if sys_os == 'Windows':
                        os.system('cls')
                    else:
                        os.system('clear')
                    if render_type == 2:
                        clear_output(True)

                    print("turn {}/{}, state {}, action, {}, reward {}, done "
                          "{}".format(step, self.max_step, state, action,
                                      reward, done))
                    self._draw(state)
                    if done and reward == REWARD_SUCCESS:
                        print("Success!")
                        time.sleep(1)
                    time.sleep(0.2)
                if done and reward == REWARD_SUCCESS:
                    n_succ += 1
                step += 1
        return float(n_succ) / N_PLAYRUN

    def display(self, policy, trial_n, playtime=1, render_type=2):
        n_succ = 0
        if render_type == 2:
            from IPython.display import clear_output

        for i in range(playtime):
            state = self.reset()
            done = False
            step = 1
            while not done:
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                state, reward, done, _ = self.step(action)
                if render_type > 0:
                    if render_type == 1:
                        sys_os = get_platform()
                        if sys_os == 'Windows':
                            os.system('cls')
                        else:
                            os.system('clear')
                    elif render_type == 2:
                        clear_output(True)

                    print("trials {}, turn {}/{}, state {}, action, {}, reward {}, done "
                          "{}".format(trial_n, step, self.max_step, state, action,
                                      reward, done))
                    self._draw(state)
                    if done and reward == REWARD_SUCCESS:
                        print("Success!")
                        time.sleep(1)
                    time.sleep(0.2)
                if done and reward == REWARD_SUCCESS:
                    n_succ += 1
                step += 1
        pass

    def _draw(self, state):
        x, y, vx, vy = state
        print(self.amap.make_draw_data(x, y))

    def save(self, Q, filenm):
        with open(filenm, 'w') as f:
            for k, v in Q.items():
                v = np.array_str(v, max_line_width=1000)
                f.write('{}\t{}\n'.format(k, v))

    def load(self, filenm):
        Q = defaultdict(lambda: np.zeros(self.action_space.n))
        with open(filenm, 'r') as f:
            for line in f:
                state, probs = line.split('\t')
                state = eval(state)
                probs = np.fromstring(probs[1:-1], sep=' ')
                Q[state] = np.array(probs)
        return Q
