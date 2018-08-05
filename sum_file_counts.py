#!/usr/bin/env python3
# Given input of lines like
# spec1d.GN0a.022.404.fits: 8
# spec1d.c1a.034.38017.fits: 7
# spec1d.E1a.012.10051.fits: 7
# Sum the counts of each file and output the summary, sorted by sum

import sys

sums = {}
for line in sys.stdin:
    filename, count = [x.strip() for x in line.split(': ')]
    sums.setdefault(filename, 0)
    sums[filename] += int(count)

for filename, count in sorted(sums.items(), key=lambda kv: kv[1], reverse=True):
    print('{}: {}'.format(filename, count))
