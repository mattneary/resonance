import sys
import json
from functools import reduce
from .resonance import resonance

text_a, text_b = [open(arg).read() for arg in sys.argv[1:]]
res_1 = resonance(text_a, text_b)
res_2 = resonance(text_b, text_a)
score = (res_1 + res_2) / 2
print(score.numpy())
