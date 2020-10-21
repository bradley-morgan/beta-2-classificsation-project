import numpy as np
# following the equstion given by: https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

def matthews_coefficient(confusion_matrix):
    TP = confusion_matrix[0, 0]
    TN = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    numer = TP * TN - FP * FN
    a = TP + FP
    b = TP + FN
    c = TN + FP
    d = TN + FN

    dinom = None
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Handle divisions by zero
        dinom = 1

    else:
        dinom = np.sqrt(a*b*c*d)

    coe = numer / dinom
    return coe



