import taichi as ti

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

x = ti.field(dtype=int, shape=(5))

@ti.kernel
def func():
    for i in range(5):
        x[i] = i

@ti.kernel
def printx():
    for i in range(5):
        print(i, x[i])

func()
printx()
alg = ti.algorithms.PrefixSumExecutor(5)
alg.run(x)
printx()
