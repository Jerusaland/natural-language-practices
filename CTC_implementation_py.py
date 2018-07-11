

import numpy as np

np.random.seed(1111)

T, V = 12, 5
m, n = 6, V

x = np.random.random([T, m])  # T x m
w = np.random.random([m, n])  # weights, m x n

def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist

def toy_nw(x):
    y = np.matmul(x, w)  # T x n 
    y = softmax(y)
    return y

y = toy_nw(x)
print(y)
print(y.sum(1, keepdims=True))


def forward(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha = np.zeros([T, L])

    # init
    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]

            a = alpha[t - 1, i] 
            if i - 1 >= 0:
                a += alpha[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha[t - 1, i - 2]

            alpha[t, i] = a * y[t, s]

    return alpha

labels = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank
alpha = forward(y, labels)
print(alpha)
p = alpha[-1, labels[-1]] + alpha[-1, labels[-2]]
print(p)


def backward(y, labels):
    T, V = y.shape
    L = len(labels)
    beta = np.zeros([T, L])

    # init
    beta[-1, -1] = y[-1, labels[-1]]
    beta[-1, -2] = y[-1, labels[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]

            a = beta[t + 1, i] 
            if i + 1 < L:
                a += beta[t + 1, i + 1]
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a += beta[t + 1, i + 2]

            beta[t, i] = a * y[t, s]

    return beta

beta = backward(y, labels)
print(beta)


def gradient(y, labels):
    T, V = y.shape
    L = len(labels)

    alpha = forward(y, labels)
    beta = backward(y, labels)
    p = alpha[-1, -1] + alpha[-1, -2]

    grad = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                grad[t, s] += alpha[t, i] * beta[t, i] 
            grad[t, s] /= y[t, s] ** 2

    grad /= p
    return grad

grad = gradient(y, labels)
print(grad)


def check_grad(y, labels, w=-1, v=-1, toleration=1e-3):
    grad_1 = gradient(y, labels)[w, v]

    delta = 1e-10
    original = y[w, v]

    y[w, v] = original + delta
    alpha = forward(y, labels)
    log_p1 = np.log(alpha[-1, -1] + alpha[-1, -2])

    y[w, v] = original - delta
    alpha = forward(y, labels)
    log_p2 = np.log(alpha[-1, -1] + alpha[-1, -2])

    y[w, v] = original

    grad_2 = (log_p1 - log_p2) / (2 * delta)
    if np.abs(grad_1 - grad_2) > toleration:
        print('[%d, %d]：%.2e' % (w, v, np.abs(grad_1 - grad_2)))

for toleration in [1e-5, 1e-6]:
    print('%.e' % toleration)
    for w in range(y.shape[0]):
        for v in range(y.shape[1]):
            check_grad(y, labels, w, v, toleration)


def gradient_logits_naive(y, labels):
    '''
    gradient by back propagation
    '''
    y_grad = gradient(y, labels)

    sum_y_grad = np.sum(y_grad * y, axis=1, keepdims=True)
    u_grad = y * (y_grad - sum_y_grad) 

    return u_grad

def gradient_logits(y, labels):
    '''
    '''
    T, V = y.shape
    L = len(labels)

    alpha = forward(y, labels)
    beta = backward(y, labels)
    p = alpha[-1, -1] + alpha[-1, -2]

    u_grad = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                u_grad[t, s] += alpha[t, i] * beta[t, i] 
            u_grad[t, s] /= y[t, s] * p

    u_grad -= y
    return u_grad

grad_l = gradient_logits_naive(y, labels)
grad_2 = gradient_logits(y, labels)

print(np.sum(np.abs(grad_l - grad_2)))


def check_grad_logits(x, labels, w=-1, v=-1, toleration=1e-3):
    grad_1 = gradient_logits(softmax(x), labels)[w, v]

    delta = 1e-10
    original = x[w, v]

    x[w, v] = original + delta
    y = softmax(x)
    alpha = forward(y, labels)
    log_p1 = np.log(alpha[-1, -1] + alpha[-1, -2])

    x[w, v] = original - delta
    y = softmax(x)
    alpha = forward(y, labels)
    log_p2 = np.log(alpha[-1, -1] + alpha[-1, -2])

    x[w, v] = original

    grad_2 = (log_p1 - log_p2) / (2 * delta)
    if np.abs(grad_1 - grad_2) > toleration:
        print('[%d, %d]：%.2e, %.2e, %.2e' % (w, v, grad_1, grad_2, np.abs(grad_1 - grad_2)))

np.random.seed(1111)
x = np.random.random([10, 10])
for toleration in [1e-5, 1e-6]:
    print('%.e' % toleration)
    for w in range(x.shape[0]):
        for v in range(x.shape[1]):
            check_grad_logits(x, labels, w, v, toleration)


ninf = -np.float('inf')

def _logsumexp(a, b):
    '''
    np.log(np.exp(a) + np.exp(b))

    '''

    if a < b:
        a, b = b, a

    if b == ninf:
        return a
    else:
        return a + np.log(1 + np.exp(b - a)) 

def logsumexp(*args):
    '''
    from scipy.special import logsumexp
    logsumexp(args)
    '''
    res = args[0]
    for e in args[1:]:
        res = _logsumexp(res, e)
    return res
	

def forward_log(log_y, labels):
    T, V = log_y.shape
    L = len(labels)
    log_alpha = np.ones([T, L]) * ninf

    # init
    log_alpha[0, 0] = log_y[0, labels[0]]
    log_alpha[0, 1] = log_y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]

            a = log_alpha[t - 1, i]
            if i - 1 >= 0:
                a = logsumexp(a, log_alpha[t - 1, i - 1])
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a = logsumexp(a, log_alpha[t - 1, i - 2])

            log_alpha[t, i] = a + log_y[t, s]

    return log_alpha

log_alpha = forward_log(np.log(y), labels)
alpha = forward(y, labels)
print(np.sum(np.abs(np.exp(log_alpha) - alpha)))


def backward_log(log_y, labels):
    T, V = log_y.shape
    L = len(labels)
    log_beta = np.ones([T, L]) * ninf

    # init
    log_beta[-1, -1] = log_y[-1, labels[-1]]
    log_beta[-1, -2] = log_y[-1, labels[-2]]

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]

            a = log_beta[t + 1, i] 
            if i + 1 < L:
                a = logsumexp(a, log_beta[t + 1, i + 1])
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a = logsumexp(a, log_beta[t + 1, i + 2])

            log_beta[t, i] = a + log_y[t, s]

    return log_beta

log_beta = backward_log(np.log(y), labels)
beta = backward(y, labels)
print(np.sum(np.abs(np.exp(log_beta) - beta)))


def gradient_log(log_y, labels):
    T, V = log_y.shape
    L = len(labels)

    log_alpha = forward_log(log_y, labels)
    log_beta = backward_log(log_y, labels)
    log_p = logsumexp(log_alpha[-1, -1], log_alpha[-1, -2])

    log_grad = np.ones([T, V]) * ninf
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                log_grad[t, s] = logsumexp(log_grad[t, s], log_alpha[t, i] + log_beta[t, i]) 
            log_grad[t, s] -= 2 * log_y[t, s]

    log_grad -= log_p
    return log_grad

log_grad = gradient_log(np.log(y), labels)
grad = gradient(y, labels)
#print(log_grad)
#print(grad)
print(np.sum(np.abs(np.exp(log_grad) - grad)))


def forward_scale(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha_scale = np.zeros([T, L])

    # init
    alpha_scale[0, 0] = y[0, labels[0]]
    alpha_scale[0, 1] = y[0, labels[1]]
    Cs = []

    C = np.sum(alpha_scale[0])
    alpha_scale[0] /= C
    Cs.append(C)

    for t in range(1, T):
        for i in range(L):
            s = labels[i]

            a = alpha_scale[t - 1, i] 
            if i - 1 >= 0:
                a += alpha_scale[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha_scale[t - 1, i - 2]

            alpha_scale[t, i] = a * y[t, s]

        C = np.sum(alpha_scale[t])
        alpha_scale[t] /= C
        Cs.append(C)

    return alpha_scale, Cs
	

labels = [0, 1, 2, 0]  # 0 for blank

alpha_scale, Cs = forward_scale(y, labels)
log_p = np.sum(np.log(Cs)) + np.log(alpha_scale[-1][labels[-1]] + alpha_scale[-1][labels[-2]])

alpha = forward(y, labels)
p = alpha[-1, labels[-1]] + alpha[-1, labels[-2]]

print(np.log(p), log_p, np.log(p) - log_p)


def backward_scale(y, labels):
    T, V = y.shape
    L = len(labels)
    beta_scale = np.zeros([T, L])

    # init
    beta_scale[-1, -1] = y[-1, labels[-1]]
    beta_scale[-1, -2] = y[-1, labels[-2]]

    Ds = []

    D = np.sum(beta_scale[-1,:])
    beta_scale[-1] /= D
    Ds.append(D)

    for t in range(T - 2, -1, -1):
        for i in range(L):
            s = labels[i]

            a = beta_scale[t + 1, i] 
            if i + 1 < L:
                a += beta_scale[t + 1, i + 1]
            if i + 2 < L and s != 0 and s != labels[i + 2]:
                a += beta_scale[t + 1, i + 2]

            beta_scale[t, i] = a * y[t, s]

        D = np.sum(beta_scale[t])
        beta_scale[t] /= D
        Ds.append(D)

    return beta_scale, Ds[::-1]

beta_scale, Ds = backward_scale(y, labels)
print(beta_scale)


def gradient_scale(y, labels):
    T, V = y.shape
    L = len(labels)

    alpha_scale, _ = forward_scale(y, labels)
    beta_scale, _ = backward_scale(y, labels)

    grad = np.zeros([T, V])
    for t in range(T):
        for s in range(V):
            lab = [i for i, c in enumerate(labels) if c == s]
            for i in lab:
                grad[t, s] += alpha_scale[t, i] * beta_scale[t, i]
            grad[t, s] /= y[t, s] ** 2

        # normalize factor
        z = 0
        for i in range(L):
            z += alpha_scale[t, i] * beta_scale[t, i] / y[t, labels[i]]
        grad[t] /= z

    return grad

labels = [0, 3, 0, 3, 0, 4, 0]  # 0 for blank
grad_1 = gradient_scale(y, labels)
grad_2 = gradient(y, labels)
print(np.sum(np.abs(grad_1 - grad_2)))


def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank     
    new_labels = [l for l in new_labels if l != blank]

    return new_labels

def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels

def greedy_decode(y, blank=0):
    raw_rs = np.argmax(y, axis=1)
    rs = remove_blank(raw_rs, blank)
    return raw_rs, rs

np.random.seed(1111)
y = softmax(np.random.random([20, 6]))
rr, rs = greedy_decode(y)
print(rr)
print(rs)


def beam_decode(y, beam_size=10):
    T, V = y.shape
    log_y = np.log(y)

    beam = [([], 0)]
    for t in range(T):  # for every timestep
        new_beam = []
        for prefix, score in beam:
            for i in range(V):  # for every state
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]

                new_beam.append((new_prefix, new_score))

        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    return beam

np.random.seed(1111)
y = softmax(np.random.random([20, 6]))
beam = beam_decode(y, beam_size=100)
for string, score in beam[:20]:
    print(remove_blank(string), score)
	

from collections import defaultdict

def prefix_beam_decode(y, beam_size=10, blank=0):
    T, V = y.shape
    log_y = np.log(y)

    beam = [(tuple(), (0, ninf))]  # blank, non-blank
    for t in range(T):  # for every timestep
        new_beam = defaultdict(lambda : (ninf, ninf))

        for prefix, (p_b, p_nb) in beam:
            for i in range(V):  # for every state
                p = log_y[t, i]

                if i == blank:  # propose a blank
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_p_b = logsumexp(new_p_b, p_b + p, p_nb + p)
                    new_beam[prefix] = (new_p_b, new_p_nb)
                    continue
                else:  # extend with non-blank
                    end_t = prefix[-1] if prefix else None

                    # exntend current prefix
                    new_prefix = prefix + (i,)
                    new_p_b, new_p_nb = new_beam[new_prefix]
                    if i != end_t:
                        new_p_nb = logsumexp(new_p_nb, p_b + p, p_nb + p)
                    else:
                        new_p_nb = logsumexp(new_p_nb, p_b + p)
                    new_beam[new_prefix] = (new_p_b, new_p_nb)

                    # keep current prefix
                    if i == end_t:
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_nb = logsumexp(new_p_nb, p_nb + p)
                        new_beam[prefix] = (new_p_b, new_p_nb)

        # top beam_size
        beam = sorted(new_beam.items(), key=lambda x : logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    return beam

np.random.seed(1111)
y = softmax(np.random.random([20, 6]))
beam = prefix_beam_decode(y, beam_size=100)
for string, score in beam[:20]:
    print(remove_blank(string), score)
	
