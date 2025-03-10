import numpy as np
from collections import defaultdict
import random
from tqdm import trange
import copy
import networkx as nx
import matplotlib.pyplot as plt
import sys
import igraph
from matplotlib import cm, colors
random.seed(42)
import seaborn as sns

class Environment:
    """Abstract Base Class for all environment classes"""
    def __init__(self, start:int, cues: list):
        self.start_state = start
        self.cue_states = cues
        self.clone_dict = dict()
        self.reverse_clone_dict = dict()







