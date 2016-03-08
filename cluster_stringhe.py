#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division


import os
import re
import sys

from itertools import combinations

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

raw = [
"i want to eat pizza",
"i want to eat cake",
"i want to eat ice cream",

"i want to drink soda",
"i want to drink tea",
"i want to drink coffee",

"i want to watch a movie",
"i want to watch a game",
"i want to watch a book",
"tostapane cannibale"
]


def to_tokens(s):
    return set(ALPHANUM_REGEX.sub(' ', s).lower().split())


def jaccard_distance(x, y):
    return 1 - (len(x['tokens'] & y['tokens']) / len(x['tokens'] | y['tokens']))


ALPHANUM_REGEX = re.compile('[\\W+]', re.UNICODE)


objs = []
distances = []

cnt = 0
for item in raw:
    objs.append({'_id':cnt, 'text':item,'tokens':to_tokens(item)})
    cnt+=1


distances = [jaccard_distance(x, y) for x, y in combinations(objs, 2)]

Z_first = linkage(distances, method='complete')

# subset = objs[:3]
# print subset

plt.figure(figsize=(10, 4))
plt.title('Z_first')
dendrogram(
    Z_first,
    orientation='left',
    leaf_font_size=8.,
)
plt.show()