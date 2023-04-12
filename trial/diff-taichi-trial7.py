import taichi as ti
import sys

ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

NUM = 7
x = ti.field(dtype=float)
loss = ti.field(dtype=float)
step = ti.field(dtype=int)
ti.root.dense(ti.i, (NUM)).place(x)
ti.root.place(loss, step)
ti.root.lazy_grad()

@ti.kernel
def initialize():
    x[0] = 2
    step[None] = 0

@ti.kernel
def func():
    while step[None] < NUM - 1:
        x[step[None] + 1] = x[step[None]] * x[step[None]]
        step[None] += 1

@ti.kernel
def func1():
    while x[NUM - 1] != 0:
        loss[None] += 1

@ti.kernel
def func2():
    ti.loop_config(serialize=True)
    for i in range(NUM - 1):
        x[i + 1] = x[i] * x[i]

@ti.kernel
def func3():
    ti.loop_config(serialize=True)
    for i in range(NUM - 1):
        x[step[None] + 1] = x[step[None]] * x[step[None]]
        step[None] += 1
        print(step[None])

def func4():
    for i in range(NUM - 1):
        func_ass(i)

@ti.kernel
def func_ass(i: int):
    x[i + 1] = x[i] + x[i]

def func5():
    i = 0
    while i < NUM - 1:
        func_ass(i)
        i += 1

def func6():
    i = 0
    while i < NUM - 1:
        func_ass(step[None])
        step[None] += 1
        i += 1

def func7():
    i = 0
    while i < NUM - 1:
        func_ass(step[None])
        step[None] += 1
        i += 1
    step[None] = 0

def func8():
    i = 0
    while i < 3:
        func_ass(step[None])
        step[None] += 1
        i += 1
    i = 0
    while i < 3:
        func_ass(step[None])
        step[None] += 1
        i += 1
    step[None] = 0

@ti.kernel
def func_ass2():
    x[step[None] + 1] = x[step[None]] + x[step[None]]

def func9():
    i = 0
    while i < NUM - 1:
        func_ass2()
        step[None] += 1
        i += 1

@ti.kernel
def func_ass3():
    i = step[None]
    x[i + 1] = x[i] + x[i]

def func10():
    i = 0
    while i < NUM - 1:
        func_ass3()
        step[None] += 1
        i += 1

@ti.kernel
def get_loss():
    loss[None] = x[NUM - 1]
    

def print_result():
    print("-------------------------------------")
    for i in range(NUM):
        print(x[i], x.grad[i])
    print(loss[None], loss.grad[None])
    print("-------------------------------------")


initialize()
with ti.ad.Tape(loss, validation=True):
    # func() error
    # func1() error
    # func2() forward OK, no backward
    # func3() error (index)
    # func4() ok
    # func5() ok
    # func6() ok
    # func7() ok!!! surprisingly
    # func8() ok
    # func9() error (index) (guess similar to func3)
    # func10() error
    get_loss()
print_result()