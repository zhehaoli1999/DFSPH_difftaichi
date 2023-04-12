import taichi as ti
import sys
sys.path.append('..')
from sph_base import *

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

NUM = 7
x = ti.field(dtype=float)
y = ti.field(dtype=float)
A = ti.field(dtype=int)
ti.root.dense(ti.i, NUM).place(x, y, A)
loss = ti.field(dtype=float, shape=())
ti.root.lazy_grad()

@ti.kernel
def initialize():
    for i in x:
        x[i] = i
        A[i] = 1 if i > 3 else (i % 2)
        loss[None] = 0

@ti.kernel
def oper():
    for i in range(NUM):
        if A[i]:
            y[i] = x[i]

@ti.kernel
def compute_loss():
    for i in range(NUM):
        loss[None] += y[i]

def print_result():
    print("-------------------------------------")
    for i in range(NUM):
        print(i, A[i], x[i], y[i])
    for i in range(NUM):
        print(i, x.grad[i], y.grad[i])
    print(loss[None])
    print("-------------------------------------")

initialize()
with ti.ad.Tape(loss=loss, validation=True):
    oper()
    compute_loss()
print_result()