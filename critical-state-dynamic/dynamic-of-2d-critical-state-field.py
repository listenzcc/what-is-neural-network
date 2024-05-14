"""
File: dynamic-of-2d-critical-state-field.py
Author: Chuncheng Zhang
Date: 2024-04-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Simulation of the dynamic of the 2d critical state field 

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-04-24 ------------------------
# Requirements and constants
import cv2
import time
import contextlib
import numpy as np

from threading import Thread
from loguru import logger


# %% ---- 2024-04-24 ------------------------
# Constants

class Simulation:
    size = (200, 200)


class BoltzmannDistribution(object):
    k = 1e-1
    temperature = 1e2
    cache = {}

    def clear_cache(self):
        # Clear the cache if k or temperature is changed
        self.cache = {}
        logger.debug('Cleared cache')

    def prob(self, energy: float):
        # Use the cache, and compute if necessary
        prob = self.cache.get(str(energy))
        return prob if prob else self._energy2prob(energy)

    def _energy2prob(self, energy: float):
        # Computation is required
        den = self.k * self.temperature
        prob = np.exp(- (energy+1e-5) / den)
        self.cache[str(energy)] = prob
        logger.debug(
            f'Computed prob: {prob} | energy: {energy}')

        return prob


sim = Simulation()
bzd = BoltzmannDistribution()

# %% ---- 2024-04-24 ------------------------
# Function and class


class HowFastIsIt(object):
    times = []

    @contextlib.contextmanager
    def timeit(self):
        try:
            tic = time.process_time()
            yield
        finally:
            t = time.process_time() - tic
            self.times.append(t)

    def report(self):
        n = len(self.times)
        mean = np.mean(self.times)
        std = np.std(self.times)
        logger.debug(
            f'Report the time cost with {n} samples, mean={mean}, std={std}')


class DynamicCriticalField(object):
    size = sim.size
    field = np.zeros(size)

    def convert_to_cv2(self):
        mat = np.concatenate([
            self.field[:, :, np.newaxis],
            self.field[:, :, np.newaxis],
            self.field[:, :, np.newaxis]
        ], axis=2)

        mat *= 255
        mat = mat.astype(np.uint8)

        # mat = cv2.resize(mat, (500, 500), interpolation=cv2.INTER_AREA)
        mat = cv2.resize(mat, (500, 500))

        return mat

    def compute_energy(self):
        # --------------------
        # Work on copy
        field = self.field.copy()

        # --------------------
        # Compute energy change,
        # the elements are the energy changes **IF** it flips
        a = field[1:-1, 1:-1]
        _energy_change = a * 0

        b = field[:-2, 1:-1]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b
        b = field[2:, 1:-1]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b
        b = field[1:-1, :-2]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b
        b = field[1:-1, 2:]
        _energy_change += 4 * a * b + 1 - 2 * a - 2 * b

        energy_change = field * 0
        energy_change[1:-1, 1:-1] = _energy_change

        '''
        Flip the values by the probability of energy changes,
        example of the energy-probability pairs are:

        - Computed prob: 0.9999 | energy: 0
        - Computed prob: 0.3678 | energy: 1
        - Computed prob: 0.1353 | energy: 2
        - Computed prob: 0.0497 | energy: 3
        - Computed prob: 0.0183 | energy: 4
        '''

        energies = [0, 1, 2, 3, 4]
        probs = [bzd.prob(e) for e in energies]
        random = np.random.random(field.shape)

        for prob, energy in zip(probs, energies):
            if energy == 0:
                # --------------------
                # Energy unchanged, flip at very low probability
                flip_prob = 1 - prob
                flip_map = (energy_change == energy) & (random < flip_prob)
                field[flip_map] = 1 - field[flip_map]
            else:
                # --------------------
                # Energy increase, hard to flip if it increases energy
                flip_prob = prob
                flip_map = (energy_change == energy) & (random < flip_prob)
                field[flip_map] = 1 - field[flip_map]

                # --------------------
                # Energy decrease, easy to flip if it decreases energy
                flip_prob = 1 - prob
                flip_map = (energy_change == -energy) & (random < flip_prob)
                field[flip_map] = 1 - field[flip_map]

        self.field = field

        return field


# %% ---- 2024-04-24 ------------------------
# Play ground
if __name__ == "__main__":
    dcf = DynamicCriticalField()

    hfii = HowFastIsIt()

    # --------------------
    # Setup image UI
    winname = 'main'
    total = int(1e4)
    frame_gap = 50  # milliseconds

    def loop():
        for j in range(total):
            with hfii.timeit():
                dcf.compute_energy()
            cv2.setWindowTitle(
                winname, f'{winname}: {j} | {total} |  {bzd.temperature} |')
            cv2.imshow(winname, dcf.convert_to_cv2())
            # cv2.waitKey(frame_gap)
            cv2.pollKey()
            time.sleep(frame_gap * 1e-3)

            if j % 100 == 0:
                hfii.report()

    Thread(target=loop, daemon=True).start()

    '''
    Control the temperature
    The critical state temperature is about 10.0 degrees
    '''
    inp = ''
    while True:
        inp = input()

        if inp == 'q':
            break

        if inp.startswith('t'):
            bzd.temperature = max(1e-1, float(inp[1:]))
            bzd.clear_cache()

    print('Done.')


# %% ---- 2024-04-24 ------------------------
# Pending


# %% ---- 2024-04-24 ------------------------
# Pending
