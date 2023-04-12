import taichi as ti
import sys
sys.path.append('..')
from sph_base import *

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

NUM = 7
x = ti.field(dtype=float, shape=(), needs_grad=True)
y = ti.field(dtype=float, shape=(), needs_grad=True)
loss = ti.field(dtype=float, shape=(), needs_grad=True)

@ti.kernel
def initialize():
    x[None] = 1
    y[None] = 2

@ti.kernel
def kernel_func1():
    loss[None] += x[None] * x[None]

@ti.kernel
def kernel_func2():
    loss[None] += x[None] * y[None]
    

def print_result():
    print("-------------------------------------")
    print(x[None], x.grad[None])
    print(y[None], y.grad[None])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")

def func1():
    kernel_func1()
    kernel_func2()

@ti.ad.grad_replaced
def func2():
    kernel_func1()
    kernel_func2()

@ti.ad.grad_for(func2)
def func2_grad():
    kernel_func1.grad()

@ti.ad.grad_replaced
def func3():
    func1()

@ti.ad.grad_for(func3)
def func3_grad():
    # func1.grad() error
    func2.grad() # okay
    # func2_grad() okay

initialize()
with ti.ad.Tape(loss, validation=True):
    func1()
print_result()

initialize()
with ti.ad.Tape(loss, validation=True):
    func2()
print_result()

initialize()
with ti.ad.Tape(loss, validation=True):
    func3()
print_result()