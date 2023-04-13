import taichi as ti

ti.init(arch=ti.cpu)
NUM_STEPS = 5
NUM_OBJECTS = 2
A = ti.field(dtype=int, shape=(NUM_OBJECTS))
A[0] = 0
A[1] = 1
# x = ti.Vector.field(3, dtype=float, shape=(NUM_STEPS, NUM_OBJECTS), needs_dual=True)
# v = ti.Vector.field(3, dtype=float, shape=(NUM_STEPS, NUM_OBJECTS), needs_dual=True)
x = ti.Vector.field(3, dtype=float, needs_dual=True)
v = ti.Vector.field(3, dtype=float, needs_dual=True)
ti.root.dense(ti.ij, (NUM_STEPS, NUM_OBJECTS)).place(x, x.dual, v, v.dual)
print(x.dual.dtype, x.dtype)

delta_v = ti.field(dtype=float, shape=(3), needs_dual=True)
delta_v.fill(0.0)
delta_v[0] = 1.0
loss = ti.field(dtype=float, shape=(), needs_dual=True)
dt = ti.field(dtype=float, shape=())
dt[None] = 4e-3

@ti.kernel
def init():
    for obj_id in range(NUM_OBJECTS):
        x[0, obj_id].fill(0.0)
        v[0, obj_id].fill(0.0)
    loss[None] = 0.0
    
    for obj_id in range(NUM_OBJECTS):
        if A[obj_id] == 1:
            for i in range(3):
                v[0, obj_id][i] += delta_v[i]

@ti.kernel
def advect(step: int):
    for obj_id in range(NUM_OBJECTS):
        if A[obj_id] == 1:
            v[step + 1, obj_id] = v[step, obj_id] + dt[None] * x[step, obj_id].dot(v[step, obj_id]) * x[step, obj_id]
            print_result()
            x[step + 1, obj_id] = x[step, obj_id] + dt[None] * v[step + 1, obj_id]
            print_result()

@ti.kernel
def compute_loss():
    loss[None] = x[NUM_STEPS - 1, 1].norm_sqr()

@ti.func
def print_result():
    print("------------------------------------------------")
    for s in range(NUM_STEPS):
        for o in range(NUM_OBJECTS):
            if A[o] == 1:
                print(f"step {s}, object {o} ----------------")
                print(f"x: {x[s,o]}, {x.dual[s,o]}")
                print(f"v: {v[s,o]}, {v.dual[s,o]}")
    print(f"loss: {loss[None]}, {loss.dual[None]}")
    print("------------------------------------------------")

def step(step):
    advect(step)

@ti.kernel
def final():
    print_result()

i = 0
with ti.ad.FwdMode(loss=loss, param=delta_v, seed=[1.0, 0.0, 0.0]):
    init()
    while i < NUM_STEPS - 1:
        step(i)
        i += 1
    compute_loss()
    final()

