import taichi as ti
import sys
sys.path.append('..')
from sph_base import *

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

x = ti.field(dtype=float, shape=(), needs_grad=True)
y = ti.field(dtype=float, shape=(), needs_grad=True)
loss = ti.field(dtype=float, shape=(), needs_grad=True)

@ti.kernel
def initialize():
    x[None] = 1
    y[None] = 2

@ti.kernel
def fail():
    i = 0
    for _ in range(5):
        i = _

@ti.kernel
def compute_loss():
    loss[None] = x[None] + y[None]

@ti.ad.grad_replaced
def func():
    compute_loss()
    fail()

@ti.ad.grad_for(func)
def func_grad():
    compute_loss.grad()

def print_result():
    print("-------------------------------------")
    print(x[None], x.grad[None])
    print(y[None], y.grad[None])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")

initialize()
fail()
with ti.ad.Tape(loss, validation=True):
    func()

print_result()