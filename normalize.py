import numpy as np

def norm(orgarr, pr, re):

    md = np.array(orgarr)
    num_inc = int(len(orgarr) * (pr / 100))
    instance_indices = np.random.choice(len(orgarr), num_inc, replace=False)
    md[instance_indices] = re
    return list(md)
