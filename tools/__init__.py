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

# def load_scene_spec(scene_name=None):
#     sub_path = self.vir_scene_path if s_type == "vir" else self.phy_scene_path
#     dirname = os.path.abspath(self.root_path + sub_path)
#     if scene_name is None:
#         target_scene_path = filedialog.askopenfilename(title='Open scene', initialdir=dirname,
#                                                        filetypes=[('xml', '*.xml'), ('All Files', '*')])
#     else:
#         target_scene_path = dirname + "\\" + scene_name + ".xml"
#     scene_dir, scene_name = os.path.split(target_scene_path)
#     f_seg = scene_name.split(".")
#     if s_type == "vir":
#         self.target_vir_scene_path = target_scene_path
#         self.target_vir_traj_path = scene_dir + "\\trajs\\" + f_seg[0] + "_simu_trajs"
#     else:
#         self.target_phy_scene_path = target_scene_path
#     xml_tree = ET.ElementTree(file=target_scene_path)
#     return xml_tree, f_seg[0]