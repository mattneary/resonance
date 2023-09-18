import sys
import json
from functools import reduce
from .resonance import resonance

if not sys.stdin.isatty():
    texts = [sys.stdin.read()]
else:
    texts = [open(arg).read() for arg in sys.argv[1:]]

resonance(*texts)
