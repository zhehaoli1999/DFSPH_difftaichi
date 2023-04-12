import taichi as ti

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

x = ti.field(dtype=int, shape=())
y = ti.field(dtype=int, shape=())

@ti.kernel
def func():
    x[None] = 3
    y[None] = ti.atomic_sub(x[None], 1) - 1

@ti.kernel
def printx():
    print(x[None], y[None])

func()
printx()
