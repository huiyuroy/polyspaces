from collections import deque
from distutils.util import strtobool

import math
import random
import pickle
from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple, List, Set, Dict, Sequence
import numpy as np

import tools.algebra as alg
import tools.geometry as geo
import tools.image as img

EPS = 1e-6
PI = np.pi
PI_2 = np.pi * 2
PI_1_2 = np.pi * 0.5
PI_1_4 = np.pi * 0.25
RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180

REV_PI = 1 / PI
REV_PI_2 = 1 / PI_2

GRID_WIDTH = 20

HUMAN_STEP = 120