import pyamg
from pyamg.gallery import poisson

A = poisson((100, 100), format='csr') # 10000 node matrix
ml = pyamg.smoothed_aggregation_solver(A, max_levels=4)
print(ml)
for i in range(len(ml.levels) - 1):
    print(f"Level {i}")
    print(f"A shape: {ml.levels[i].A.shape}")
    print(f"P shape: {ml.levels[i].P.shape}")
    print(f"R shape: {ml.levels[i].R.shape}")
    
