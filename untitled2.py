# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:30:14 2019

@author: user
"""

from tqdm import tqdm
from tqdm import trange
bar = tqdm(["a", "b", "c", "d"])
for char in bar:
    bar.set_description("Processing %s" % char)