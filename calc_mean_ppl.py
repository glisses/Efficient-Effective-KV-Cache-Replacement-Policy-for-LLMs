import numpy as np

def calc_mean_ppl(path):
    with open(path, "r") as f:
        nlls = [float(line.strip()) for line in f]
    return np.exp(np.mean(nlls))

# print(calc_mean_ppl("outputs/debug/log.txt"))
print(calc_mean_ppl("outputs/debug/log_4_2048.txt"))