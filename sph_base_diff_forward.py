from matplotlib.pyplot import axis
import taichi as ti
import numpy as np
from particle_system_diff_forward import ParticleSystem


@ti.func
def quaternion_multiply(a: ti.types.vector(4, float), b: ti.types.vector(4, float)) -> ti.types.vector(4, float):
    return ti.Vector([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]
    ])

@ti.func
def vec32quaternion(a: ti.types.vector(3, float)) -> ti.types.vector(4, float):
    return ti.Vector([0.0, a[0], a[1], a[2]])

@ti.func
def quaternion2rotation_matrix(a: ti.types.vector(4, float)) -> ti.types.matrix(3, 3, float):
    # follow Eigen Quaternion
    tx = 2.0 * a[1]
    ty = 2.0 * a[2]
    tz = 2.0 * a[3]
    twx = tx * a[0]
    twy = ty * a[0]
    twz = tz * a[0]
    txx = tx * a[1]
    txy = ty * a[1]
    txz = tz * a[1]
    tyy = ty * a[2]
    tyz = tz * a[2]
    tzz = tz * a[3]
    return ti.Matrix(
        [[1.0 - (tyy + tzz), txy - twz, txz + twz],
        [txy + twz, 1.0 - (txx + tzz), tyz - twx],
        [txz - twy, tyz + twx, 1.0 - (txx + tyy)]]
    )

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system: ParticleSystem):
        self.ps = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.ps.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        self.g = np.array(self.ps.cfg.get_cfg("gravitation"))

        self.viscosity = 0.01  # viscosity

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.ps.cfg.get_cfg("density0")

        self.dt = ti.field(float, shape=())
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

        self.inv_dt = ti.field(float, shape=())
        self.inv_dt[None] = 1.0 / self.dt[None]
        self.inv_dt2 = ti.field(float, shape=())
        self.inv_dt2[None] = self.inv_dt[None] * self.inv_dt[None]

        self.step_num = 0
        self.iter_num = ti.field(int, shape=(self.ps.steps))
        self.divergence_iter_num = ti.field(int, shape=(self.ps.steps))
        self.pressure_iter_num = ti.field(int, shape=(self.ps.steps))
        self.current_iter = 0

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def viscosity_force(self, step, iter, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[step, iter, p_i] -
                self.ps.v[step, iter, p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.ps.m[step, p_j] / (self.ps.density[step, p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    def initialize_from_restart(self):
        self.initialize_rigid_info()
        self.initialize_rigid_particle_info()

    def initialize(self):
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_mass_info(r_obj_id)
        self.initialize_fluid_particle_info()
        self.initialize_from_restart()


    @ti.kernel
    def initialize_rigid_info(self):
        for r_obj_id in range(self.ps.num_objects):
            if self.ps.is_rigid[r_obj_id] == 1:

                self.ps.rigid_x[0, r_obj_id] = self.ps.rigid_rest_cm[r_obj_id]
                self.ps.rigid_quaternion[0, r_obj_id].fill(0.0)
                if r_obj_id == 1:
                    for i in range(3):
                        self.ps.rigid_v[0, 1][i] = self.ps.rigid_v0[1][i] + self.ps.rigid_adjust_v[i]
                        self.ps.rigid_omega[0, 1][i] = self.ps.rigid_omega0[1][i] + self.ps.rigid_adjust_omega[i]
                else:
                    self.ps.rigid_v[0, r_obj_id] = self.ps.rigid_v0[r_obj_id]
                    self.ps.rigid_omega[0, r_obj_id] = self.ps.rigid_omega0[r_obj_id]
                
                self.ps.rigid_force[0, r_obj_id].fill(0.0)
                self.ps.rigid_torque[0, r_obj_id].fill(0.0)
                
                R = quaternion2rotation_matrix(self.ps.rigid_quaternion[0, r_obj_id])
                self.ps.rigid_inertia[0, r_obj_id] = R @ self.ps.rigid_inertia0[r_obj_id] @ R.transpose()
                self.ps.rigid_inv_inertia[0, r_obj_id] = self.ps.rigid_inertia[0, r_obj_id].inverse()


    @ti.kernel
    def initialize_fluid_particle_info(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.input_material[p_i] == self.ps.material_fluid:
                self.ps.init_temp_x[p_i] = self.ps.input_x[p_i]
                self.ps.init_temp_v[p_i] = self.ps.input_v[p_i]


    @ti.kernel
    def initialize_rigid_particle_info(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.input_material[p_i] == self.ps.material_solid:
                r = self.ps.input_object_id[p_i]
                x_rel = self.ps.input_x[p_i] - self.ps.rigid_rest_cm[r]
                self.ps.init_temp_x[p_i] = self.ps.rigid_x[0, r] + quaternion2rotation_matrix(self.ps.rigid_quaternion[0, r]) @ x_rel
                self.ps.init_temp_v[p_i] = self.ps.rigid_v[0, r] + self.ps.rigid_omega[0, r].cross(x_rel)


    @ti.kernel
    def compute_rigid_mass_info(self, object_id: int):
        sum_m = 0.0
        sum_inertia = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dt=float)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.input_object_id[p_i] == object_id:
                mass = self.ps.input_m[p_i]
                sum_m += mass
                r = self.ps.input_x[p_i] - self.ps.rigid_rest_cm[object_id]
                sum_inertia += mass * (r.dot(r) * ti.Matrix.identity(ti.f32, 3) - r.outer_product(r))
        self.ps.rigid_mass[object_id] = sum_m
        self.ps.rigid_inertia0[object_id] = sum_inertia
        self.ps.rigid_inv_mass[object_id] = 1.0 / sum_m

    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)

    @ti.kernel
    def compute_static_boundary_volume(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i, step):
                delta = self.cubic_kernel(0.0)
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_boundary_volume_task, delta)
                self.ps.m_V[step, p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, step, iter, p_i, p_j, delta: ti.template()):
        if self.ps.material[step, p_j] == self.ps.material_solid:
            delta += self.cubic_kernel((self.ps.x[step, p_i] - self.ps.x[step, p_j]).norm())


    @ti.kernel
    def compute_moving_boundary_volume(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i, step):
                delta = self.cubic_kernel(0.0)
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_boundary_volume_task, delta)
                self.ps.m_V[step, p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, step, iter, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[step, iter, p_i] -= (
            1.0 + c_f) * self.ps.v[step, iter, p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, step: int, iter: int, particle_type:int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == particle_type and self.ps.is_dynamic[step, p_i]: 
                pos = self.ps.x[step, p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[step, p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[step, p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[step, p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[step, p_i][1] = self.ps.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(step, iter,
                            p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self, step: int, iter: int, particle_type: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == particle_type and self.ps.is_dynamic[step, p_i]:
                pos = self.ps.x[step, p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[step, p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[step, p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[step, p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[step, p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[step, p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[step, p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(step, iter,
                            p_i, collision_normal / collision_normal_length)


    @ti.kernel
    def solve_rigid_body(self, step: int):
        for r_obj_id in range(self.ps.num_objects):
            if self.ps.is_rigid[r_obj_id] == 1:
                self.ps.rigid_force[step, r_obj_id] += self.ps.rigid_mass[r_obj_id] * ti.Vector(self.g)

                self.ps.rigid_v[step + 1, r_obj_id] = self.ps.rigid_v[step, r_obj_id] + self.dt[None] * self.ps.rigid_force[step, r_obj_id] / self.ps.rigid_mass[r_obj_id]
                self.ps.rigid_force[step + 1, r_obj_id].fill(0.0)
                self.ps.rigid_x[step + 1, r_obj_id] = self.ps.rigid_x[step, r_obj_id] + self.dt[None] * self.ps.rigid_v[step + 1, r_obj_id]
                self.ps.rigid_omega[step + 1, r_obj_id] = self.ps.rigid_omega[step, r_obj_id] + self.dt[None] * self.ps.rigid_inv_inertia[step, r_obj_id] @ self.ps.rigid_torque[step, r_obj_id]
                self.ps.rigid_torque[step, r_obj_id].fill(0.0)
                self.ps.rigid_quaternion[step, r_obj_id] += self.dt[None] * 0.5 * quaternion_multiply(vec32quaternion(self.ps.rigid_omega[step, r_obj_id]), self.ps.rigid_quaternion[step, r_obj_id])
                self.ps.rigid_quaternion[step + 1, r_obj_id] = self.ps.rigid_quaternion[step, r_obj_id].normalized()
                R = quaternion2rotation_matrix(self.ps.rigid_quaternion[step + 1, r_obj_id])
                self.ps.rigid_inertia[step + 1, r_obj_id] = R @ self.ps.rigid_inertia0[r_obj_id] @ R.transpose()
                self.ps.rigid_inv_inertia[step + 1, r_obj_id] = self.ps.rigid_inertia[step + 1, r_obj_id].inverse()
            
    @ti.kernel
    def update_rigid_particle_info(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i, step):
                r = self.ps.object_id[step, p_i]
                x_rel = self.ps.x_0[step, p_i] - self.ps.rigid_rest_cm[r]
                self.ps.x_buffer[step, p_i] = self.ps.rigid_x[step + 1, r] + quaternion2rotation_matrix(self.ps.rigid_quaternion[step + 1, r]) @ x_rel
                self.ps.v[step, iter, p_i] = self.ps.rigid_v[step + 1, r] + self.ps.rigid_omega[step + 1, r].cross(x_rel)

    def step(self, step):
        print(f"------------step {step}------------")
        last_iter = 0
        if step != 0:
            last_iter = self.iter_num[step - 1]
        self.step_num = step
        self.iter_num[step] = 0
        self.ps.initialize_particle_system(step)
        self.ps.counting_sort(step, last_iter)
        if step == 0:
            self.compute_static_boundary_volume(step, 0)
        self.compute_moving_boundary_volume(step, 0)
        self.substep()
        self.solve_rigid_body(step)
        self.update_rigid_particle_info(step, self.iter_num[step])
        if self.ps.dim == 2:
            self.enforce_boundary_2D(step, self.iter_num[step], self.ps.material_fluid)
        elif self.ps.dim == 3:
            self.enforce_boundary_3D(step, self.iter_num[step], self.ps.material_fluid)
    

    def end(self, step):
        return step >= self.ps.steps - 1
    
    
    @ti.kernel
    def update(self, lr: float, index: int):
        print("v grad ", index, self.ps.loss.dual[None])
        self.ps.rigid_adjust_v[index] -= self.ps.loss.dual[None] * lr
    
    @ti.kernel
    def compute_loss(self, step: int):
        self.ps.loss[None] = (self.ps.rigid_x[step, 1] - ti.Vector([1.0, 1.0, 1.0])).norm_sqr()