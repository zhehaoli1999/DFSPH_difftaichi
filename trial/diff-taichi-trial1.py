import taichi as ti
import sys
sys.path.append('..')
from sph_base import *

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

NUM = 7
x = ti.field(dtype=float)
A = ti.field(dtype=int)
ti.root.dense(ti.i, NUM).place(x, A)
loss = ti.field(dtype=float, shape=())
# ti.root.lazy_grad()

@ti.kernel
def initialize():
    for i in x:
        x[i] = i
        A[i] = 1 if i > 3 else (i % 2)

@ti.kernel
def before(ret: ti.template()):
    pass

@ti.kernel
def after(ret: ti.template()):
    loss[None] = ret.sum / ret.num

@ti.kernel
def compute_loss():
    ret = ti.Struct(num = 0, sum = 0.0)
    compute_loss_impl(ret)
    loss[None] = ret.sum / ret.num
    

@ti.func
def compute_loss_impl(ret: ti.template()):
    for i in x:
        if A[i] == 1:
            ret.num += 1
            ret.sum += x[i]

def print_result():
    print("-------------------------------------")
    for i in range(NUM):
        print(i, A[i], x[i])
    # for i in range(NUM):
    #     print(i, x.grad[i])
    print(loss[None])
    print("-------------------------------------")

a = ti.Struct(num = 0, sum = 0.0)
print_result()
initialize()
print_result()
compute_loss()
print_result()
# loss.grad[None] = 1
# print_result()
# compute_loss.grad()
# print_result()