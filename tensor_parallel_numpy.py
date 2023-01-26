#!/usr/bin/env python3.11

import numpy as np
from multiprocessing import Pool

"""
This is tensor parallelism, as described here: 
https://openai.com/blog/techniques-for-training-large-neural-networks/

I believe the below are examples of master-worker AllReduce.
The "workers", here are just processes created via a multiprocessing Pool,
each return all of their results to a "master", in this case simply
the main process, which aggregates them all before proceeding to the next
step.

See here for various kinds of AllReduce: 
https://id2223kth.github.io/slides/2022/12_distributed_dl.pdf
"""

# Shard A by row, don't shard B
# Aggregate by row
def test_row_sharding(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    This shards A by row, but doesn't shard B.
    Every worker has a full copy of B.
    Each worker produces all columns and rows 0..n/(num_workers) of the result.
    Then we aggregate the results by row.
    """
    # Divide A into sub matrices
    A1, A2, A3, A4 = np.array_split(A, 4)
    A_gathered = np.concatenate((A1, A2, A3, A4))
    np.testing.assert_equal(A_gathered, A)

    # Create a pool of 4 worker processes
    with Pool(4) as p:
        # Perform dot product of each submatrix of A with B in parallel
        C1, C2, C3, C4 = p.starmap(np.dot, [(A1, B), (A2, B), (A3, B),
                                            (A4, B)])

    # Combine the results
    C = np.concatenate((C1, C2, C3, C4), axis=0)
    return C

def test_row_and_column_sharding(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    This shards A by row and B by column.
    Every worker has a shard of A and a shard of B.
    We aggregate the results by column.
    Then we aggregate the results by row.
    Example:
    A = [[1, 2],
         [3, 4]]
    B = [[2, 3,
         [4, 5]]]
    A1 = [[1, 2]]
    A2 = [[3, 4]]
    B1 = [[2],
          [4]]
    B2 = [[3],
          [5]]
    A1 . B1 = [[10]]
    A1 . B2 = [[13]]
    A2 . B1 = [[22]]
    A2 . B2 = [[29]]
    C = [[10, 13],
         [22, 29]]
    """
    # Divide A and B into sub matrices
    A1, A2, A3, A4 = np.array_split(A, 4, axis=0)
    B1, B2, B3, B4 = np.array_split(B, 4, axis=1)

    # Create a pool of 4 worker processes
    with Pool(4) as p:
        # Perform dot product of each submatrices of A and B in parallel
        C11, C12, C13, C14 = p.starmap(np.dot, [(A1, B1), (A1, B2), (A1, B3), (A1, B4)])
        C21, C22, C23, C24 = p.starmap(np.dot, [(A2, B1), (A2, B2), (A2, B3), (A2, B4)])
        C31, C32, C33, C34 = p.starmap(np.dot, [(A3, B1), (A3, B2), (A3, B3), (A3, B4)])
        C41, C42, C43, C44 = p.starmap(np.dot, [(A4, B1), (A4, B2), (A4, B3), (A4, B4)])

    C1 = np.concatenate((C11, C12, C13, C14), axis=1)
    C2 = np.concatenate((C21, C22, C23, C24), axis=1)
    C3 = np.concatenate((C31, C32, C33, C34), axis=1)
    C4 = np.concatenate((C41, C42, C43, C44), axis=1)
    C = np.concatenate((C1, C2, C3, C4), axis=0)
    return C

if __name__ == "__main__":
    A = np.random.random((10000, 10000))
    B = np.random.random((10000, 1000))
    C_truth = np.dot(A, B)

    print("Testing sharding A by row...")
    C = test_row_sharding(A, B)
    np.testing.assert_almost_equal(C, C_truth)

    print("Testing sharding A by row and B by col...")
    C = test_row_and_column_sharding(A, B)
    np.testing.assert_almost_equal(C, C_truth)
    print("Success!")
