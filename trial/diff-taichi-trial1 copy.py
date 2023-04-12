import taichi as ti
import sys
sys.path.append('..')
ti.init(arch=ti.gpu, device_memory_fraction=0.5)

NUM = 7
x = ti.field(dtype=float)
A = ti.field(dtype=int)
ti.root.dense(ti.i, NUM).place(x, A)
loss = ti.field(dtype=float, shape=())
num = ti.field(dtype=int, shape=())
sum = ti.field(dtype=float, shape=())
ti.root.lazy_grad()

@ti.kernel
def initialize():
    for i in x:
        x[i] = i
        A[i] = 1 if i > 3 else (i % 2)

@ti.kernel
def before():
    sum[None] = 0.
    num[None] = 0

@ti.kernel
def after():
    loss[None] = sum[None] / num[None]

@ti.kernel
def compute_loss():
    for i in x:
        if A[i] == 1:
            num[None] += 1
            sum[None] += x[i]

initialize()
before()
compute_loss()
after()
loss.grad[None] = 1
after.grad()
compute_loss.grad()
before.grad()



def print_result():
    print("-------------------------------------")
    for i in range(NUM):
        print(i, A[i], x[i])
    for i in range(NUM):
        print(i, x.grad[i])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")
print_result()
