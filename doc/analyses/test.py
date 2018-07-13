
import sys
sys.path.append("../../examples")
import data

exp1 = data.exp(10000, seed=1) # data sample 1 with 10k events
mc1  = data.MC(500000, seed=2) # Monte Carlo sample 1 with 500k events
livetime1 = 100.               # livetime for sample 1 in days

