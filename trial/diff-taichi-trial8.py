import taichi as ti
import sys

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

NUM = 7
x = ti.field(dtype=float)
y = ti.field(dtype=float)
ti.root.dense(ti.i, NUM).place(x, y)
loss = ti.field(dtype=float, shape=())
ti.root.lazy_grad()

@ti.kernel
def initialize():
    x[0] = 1


@ti.kernel
def func1(i: int):
    y[i] = x[i] * 2
    
@ti.kernel
def func2(i: int):
    x[i + 1] = y[i] + x[i]

@ ti.kernel
def get_loss():
    loss[None] = y[NUM - 1]

def func():
    for i in range(NUM - 1):
        func1(i)
        func2(i)
    func1(NUM - 1)

def func_grad():
    func1.grad(NUM - 1)
    print_result()
    for i in range(NUM - 1):
        func2.grad(NUM - 2 - i)
        print_result()
        func1.grad(NUM - 2 - i)
        print_result()

def print_result():
    print("-------------------------------------")
    for i in range(NUM):
        print(i, x[i], y[i])
    for i in range(NUM):
        print(i, x.grad[i], y.grad[i])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")

# initialize()
# with ti.ad.Tape(loss, validation=True):
#     func()
#     get_loss()
# print_result()

@ti.kernel
def initialize_grad():
    for i in range(NUM):
        x.grad[i] = 0
        y.grad[i] = i

initialize()
initialize_grad()
ti.ad.clear_all_gradients()
func()
get_loss()
loss.grad[None] = 1
print_result()
get_loss.grad()
print_result()
func_grad()
