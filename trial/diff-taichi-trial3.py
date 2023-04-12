import taichi as ti

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

STEPS = 3
NUM = 3
x = ti.field(dtype=float)
y = ti.field(dtype=float)
ti.root.dense(ti.ij, (STEPS, NUM)).place(x, y)
loss = ti.field(dtype=float, shape=())
ti.root.lazy_grad()

@ti.kernel
def compute_loss():
    loss[None] = y[STEPS - 1, NUM - 1]

@ti.kernel
def func(s: int):
    for i in range(NUM):
        if s > 0:
            y[s, i] = x[s, i] + y[s - 1, i]
        else:
            y[s, i] = x[s, i]

@ti.kernel
def initialize():
    for s in range(STEPS):
        for i in range(NUM):
            x[s, i] = i + 1
            y[s, i] = 0
            x.grad[s, i] = 0
            y.grad[s, i] = 0
            loss[None] = 0
            loss.grad[None] = 0

def print_result():
    print("-------------------------------------")
    for s in range(STEPS):
        for i in range(NUM):
            print(s, i, x[s, i], y[s, i])
    for s in range(STEPS):
        for i in range(NUM):
            print(s, i, x.grad[s, i], y.grad[s, i])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")

initialize()
print_result()
for s in range(STEPS):
    func(s)
print_result()
compute_loss()
print_result()
loss.grad[None] = 1
print_result()
compute_loss.grad()
print_result()
for s in range(STEPS - 1, -1, -1):
    func.grad(s)
print_result()

initialize()
with ti.ad.Tape(loss=loss):
    for s in range(STEPS):
        func(s)
    compute_loss()
print_result()
