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

@ti.func
def test(ret: ti.template()):
    ret.x += x[None] * ret.y

@ti.kernel
def compute_loss():
    ret = ti.Struct(x=0.0, y=y[None])
    test(ret)
    loss[None] = ret.x

def func():
    compute_loss()


def print_result():
    print("-------------------------------------")
    print(x[None], x.grad[None])
    print(y[None], y.grad[None])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")

initialize()
with ti.ad.Tape(loss, validation=True):
    func()

print_result()