import taichi as ti
import sys
sys.path.append('..')
from sph_base import *

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

NUM = 7
x = ti.field(dtype=float, shape=(), needs_grad=True)
y = ti.field(dtype=float, shape=())
loss = ti.field(dtype=float, shape=(), needs_grad=True)

@ti.kernel
def initialize():
    x[None] = 1
    y[None] = 2

@ti.kernel
def compute_loss1():
    x[None] += 2
    loss[None] = x[None] * x[None]
    loss[None] = 2 * x[None] * x[None] + y[None]

def print_result():
    print("-------------------------------------")
    print(x[None], x.grad[None])
    print(y[None])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")

initialize()

with ti.ad.Tape(loss, validation=True):
    compute_loss1()

print_result()