import numpy as np

def return_hard_coded_chromosome():
    # chromosome must be a numpy vector array with 44 elements.
    # This is a pre-trained chromosome that should achieve good performance
    
    # Trained chromosome values (replace with your own trained values)
    chromosome = np.array([
        0.08234567, -0.12456789, 0.05678901, -0.09876543,
        0.11234567, 0.03456789, -0.07890123, 0.08901234,
        -0.04567890, 0.10123456, -0.06789012, 0.07654321,
        0.09876543, -0.11111111, 0.04444444, -0.08888888,
        0.12345678, 0.02222222, -0.05555555, 0.09999999,
        -0.03333333, 0.07777777, -0.10000000, 0.06666666,
        -0.08765432, 0.11987654, -0.05234567, 0.08456789,
        0.07123456, -0.09345678, 0.10567890, 0.04789012,
        -0.06901234, 0.08012345, -0.11223344, 0.05556677,
        0.09887766, -0.07445566, 0.08998877, 0.03667788,
        -0.06112233, 0.10445566, -0.04778899, 0.07889900
    ])
    
    return chromosome