import torch.nn as nn
import torch
import numpy as np
import math
import code
import time
torch.manual_seed(42)


def torch_softmax(a):
    m = nn.Softmax(dim=0)
    output = m(a)
    return output
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
    ms = ms.clone().detach()
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
    # print(ms)
    return ms


def safe_softmax_online(ms):
    ms = ms.clone().detach()
    assert len(ms.shape) == 2, "input should be a matrix"
    for k in range(len(ms)):
        a = ms[k]

        # find maximum
        m = float('-inf')
        cm = float('-inf')
        s = 0
        for i in range(len(a)):
            cm = max(a[i], m)
            s = s * math.exp(m - cm) + math.exp(a[i] - cm)
            m = cm

        # device each element by division factor
        for i in range(len(a)):
            a[i] = math.exp(a[i] - m) / s
    # print(ms)
    return ms


if __name__ == "__main__":
    a = torch.arange(3).float()
    # print('Input', a)
    # torch_softmax(a)
    # softmax_numpy(a.numpy())
    # softmax_scratch(a.numpy())
    # safe_softmax_scratch(a.numpy())

    a = torch.arange(6000).reshape(200, 30).float()
    # a = torch.rand(200, 300).float()
    tik = time.time()
    out1 = safe_softmax(a)
    tok = time.time()
    print(f"time taken by non online: {tok-tik}")
    tik = time.time()
    out2 = safe_softmax_online(a)
    tok = time.time()
    print(f"time taken by online: {tok-tik}")

    sm = nn.Softmax(dim=1)
    are_close = torch.allclose(out1, sm(a.clone().detach()), rtol=1e-3, atol=1e-5)
    print(are_close)
