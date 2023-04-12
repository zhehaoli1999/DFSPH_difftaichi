import taichi as ti
import sys
sys.path.append('..')
from sph_base import *

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

NUM = 3
x = ti.Vector.field(n=3, dtype=float)
y = ti.Vector.field(n=4, dtype=float)
A = ti.field(dtype=int)
ti.root.dense(ti.i, NUM).place(x, y, A)
loss = ti.field(dtype=float, shape=())
ti.root.lazy_grad()

@ti.kernel
def compute_loss():
    loss[None] = y[1].norm_sqr()

@ti.kernel
def func():
    for i in x:
        y[i] = vec32quaternion(x[A[i]])

@ti.kernel
def initialize():
    for i in x:
        A[i] = (i + 1) % NUM
        x[i] = ti.Vector([i, i + 1.0, i + 2.0], dt=float)
        y[i].fill(0.0)

def print_result():
    print("-------------------------------------")
    for i in range(NUM):
        print(i, A[i], x[i], y[i])
    for i in range(NUM):
        print(i, x.grad[i], y.grad[i])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")

initialize()
print_result()
func()
print_result()
compute_loss()
print_result()
loss.grad[None] = 1
print_result()
compute_loss.grad()
print_result()
func.grad()
print_result()