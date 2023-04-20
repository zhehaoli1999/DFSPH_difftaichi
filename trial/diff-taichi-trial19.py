import taichi as ti

ti.init(arch=ti.cpu)

x = ti.field(dtype=float, needs_dual=True)
y = ti.field(dtype=float, needs_dual=True)
z = ti.field(dtype=float, needs_dual=True)
ti.root.place(x, x.dual, y, y.dual, z, z.dual)

loss = ti.field(dtype=float, shape=(), needs_dual=True)

@ti.kernel
def init():
    x[None] = 1.0
    y[None] = 0.0
    z[None] = 0.0
    loss[None] = 0.0

@ti.kernel
def func():
    print_result()
    z[None] = x[None]
    print_result()
    x[None] = z[None] + z[None] * y[None]
    print_result()

@ti.kernel
def get_loss():
    print_result()
    loss[None] = x[None] + y[None]
    print_result()

@ti.func
def print_result():
    print("------------------------------------------------")
    print(f"x: {x[None]}, {x.dual[None]}")
    print(f"y: {y[None]}, {y.dual[None]}")
    print(f"z: {z[None]}, {z.dual[None]}")
    print(f"loss: {loss[None]}, {loss.dual[None]}")
    print("------------------------------------------------")

init()
with ti.ad.FwdMode(loss=loss, param=x):
    func()
    get_loss()