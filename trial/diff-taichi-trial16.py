import taichi as ti
ti.init()
N_param = 2
N_loss = 5
x = ti.field(dtype=ti.f32, shape=N_param, needs_dual=True)
y = ti.field(dtype=ti.f32, shape=N_loss, needs_dual=True)

@ti.kernel
def compute_y():
    for i in range(N_loss):
        for j in range(N_param):
            y[i] += i * ti.sin(x[j])

# Compute derivatives with respect to x_0
# `seed` is required if `param` is not a scalar field
with ti.ad.FwdMode(loss=y, param=x, seed=[1.0, 0.0]):
    compute_y()
print('dy/dx_0 =', y.dual, ' at x_0 =', x[0])

# Compute derivatives with respect to x_1
# `seed` is required if `param` is not a scalar field
with ti.ad.FwdMode(loss=y, param=x, seed=[0.0, 1.0]):
    compute_y()
print('dy/dx_1 =', y.dual, ' at x_1 =', x[1])