import torch.nn as nn
import torch
import numpy as np
import math
import code


def torch_softmax(a):
    m = nn.Softmax(dim=0)
    output = m(a)
    print(output.numpy())


def softmax_numpy(a):
    result = np.exp(a) / sum(np.exp(a))
    print(result)


def softmax_scratch(a):
    a = a.copy()
    # find divisible factor
    s = 0
    for i in range(len(a)):
        s += math.exp(a[i])

    # device each element by division factor
    for i in range(len(a)):
        a[i] = math.exp(a[i]) / s
    print(a)


def safe_softmax_scratch(a):
    assert len(a.shape) == 1, "a can't be a matrix"
    # find maximum
    m = float('-inf')
    for i in range(len(a)):
        m = max(a[i], m)

    # find divisible factor
    s = 0
    for i in range(len(a)):
        s += math.exp(a[i] - m)

    # device each element by division factor
    for i in range(len(a)):
        a[i] = math.exp(a[i] - m) / s
    print(a)


def safe_softmax(ms):
    assert len(ms.shape) == 2, "input should be a matrix"
    for i in range(len(ms)):
        a = ms[i]

        # find maximum
        m = float('-inf')
        for i in range(len(a)):
            m = max(a[i], m)

        # find divisible factor
        s = 0
        for i in range(len(a)):
            s += math.exp(a[i] - m)

        # device each element by division factor
        for i in range(len(a)):
            a[i] = math.exp(a[i] - m) / s
    print(ms)


if __name__ == "__main__":
    a = torch.arange(3).float()
    # print('Input', a)
    # torch_softmax(a)
    # softmax_numpy(a.numpy())
    # softmax_scratch(a.numpy())
    # safe_softmax_scratch(a.numpy())

    a = torch.arange(6).reshape(2, 3).float()
    safe_softmax(a)
