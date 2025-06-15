import os
import sys
import math
import random
import time
import operator
import pickle

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as mb
from tkinter import filedialog

EPS = 1e-6
PI = np.pi
PI_2 = np.pi * 2
PI_1_2 = np.pi * 0.5
PI_1_4 = np.pi * 0.25
RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180
