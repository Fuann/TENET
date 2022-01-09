#!/bin/python

import sys
from pesq import pesq
from scipy.io import wavfile

ref=sys.argv[1]
deg=sys.argv[2]

rate, ref = wavfile.read(ref)
rate, deg = wavfile.read(deg)
print("%.3f" %(pesq(rate, ref, deg, 'wb')))

sys.exit(0)
