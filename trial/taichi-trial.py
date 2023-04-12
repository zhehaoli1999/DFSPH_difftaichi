import taichi as ti

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

x = ti.Vector.field(3, dtype=float, shape=())

@ti.kernel
def func():
    x[None] = ti.Vector([1.3, 1.5, 1.7])
    print(x[None].cast(int))

func()