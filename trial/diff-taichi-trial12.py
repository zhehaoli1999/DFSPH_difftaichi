import taichi as ti

ti.init(arch=ti.cpu, debug=True)

NUM = 5
x = ti.field(dtype=float)
y = ti.field(dtype=float)
ti.root.dense(ti.i, NUM).place(x, y)
loss = ti.field(dtype=float, shape=())
ti.root.lazy_grad()

@ti.kernel
def initialize():
    for i in x:
        x[i] = i
        y[i] = 0
    loss[None] = 0

@ti.kernel
def compute_loss():
    for i in range(NUM):
        l = 0.0
        for j in range(i):
            for k in range(j):
                l += x[j] * x[k]
        y[i] = l
            
    for i in range(NUM):
        loss[None] += y[i]

def print_result():
    print("-------------------------------------")
    for i in range(NUM):
        print(i, x[i], y[i])
    for i in range(NUM):
        print(i, x.grad[i], y.grad[i])
    print(loss[None])
    print("-------------------------------------")

initialize()
with ti.ad.Tape(loss=loss, validation=True):
    compute_loss()
print_result()