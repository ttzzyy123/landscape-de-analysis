
import numpy as np
import ioh

def get_bbob_function(fid1, iid1, fid2, iid2, dim, alpha):
    f1 = ioh.get_problem(fid1, iid1, dim)
    f2 = ioh.get_problem(fid2, iid2, dim)

    o1 = f1.optimum.y
    o2 = f2.optimum.y
    return lambda x : np.exp(alpha * np.log(np.clip(f1(x) - o1, 1e-12, 1e12)) + (1-alpha) * np.log(np.clip(f2(x - f1.optimum.x + f2.optimum.x) - o2, 1e-12, 1e12)))
