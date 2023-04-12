import taichi as ti

ti.init(print_ir=True)

N = 15
a = ti.field(ti.f32, shape=N, needs_grad=True)
f = ti.field(ti.f32, shape=N, needs_grad=True)

@ti.kernel
def add():
    for i in range(N):
        p = a[i]
        q = 0.0
        for j in range(5):
            q += p
        f[i] = q

add()

for i in range(N):
    f.grad[i] = 1

add.grad()

for i in range(N):
    print(a.grad[i])