from matplotlib.pyplot import axis
import taichi as ti
import numpy as np
from particle_system_diff import ParticleSystem


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

        self.step_num = ti.field(int, shape=())
        self.iter_num = ti.field(int, shape=())
        self.step_num[None] = 0
        self.iter_num[None] = 0

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
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[self.step_num[None], self.iter_num[None], p_i] -
                self.ps.v[self.step_num[None], self.iter_num[None], p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.ps.m[self.step_num[None], p_j] / (self.ps.density[self.step_num[None], p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    def initialize(self):
        self.ps.initialize_particle_system(0, 0)
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        self.initialize_rigid_info()
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_mass_info(r_obj_id)
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()

    @ti.kernel
    def initialize_rigid_info(self):
        # call in initialization after compute_rigid_rest_cm
        for r_obj_id in range(self.ps.num_objects):
            # velocities and angular velocities have already been initialized in particle system
            self.ps.rigid_x[0, r_obj_id] = self.ps.rigid_rest_cm[r_obj_id]
            self.ps.rigid_quaternion[0, r_obj_id] = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.ps.rigid_force[0, r_obj_id].fill(0.0)
            self.ps.rigid_torque[0, r_obj_id].fill(0.0)

    @ti.kernel
    def compute_rigid_mass_info(self, object_id: int):
        sum_m = 0.0
        sum_inertia = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dt=float)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.object_id[0, p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[0, p_i]
                sum_m += mass
                r = self.ps.x[0, p_i] - self.ps.rigid_x[0, object_id]
                sum_inertia += mass * (r.dot(r) * ti.Matrix.identity(ti.f32, 3) - r.outer_product(r))
        self.ps.rigid_mass[object_id] = sum_m
        self.ps.rigid_inertia0[object_id] = sum_inertia
        self.ps.rigid_inertia[0, object_id] = sum_inertia
        self.ps.rigid_inv_mass[object_id] = 1.0 / sum_m
        self.ps.rigid_inv_inertia[0, object_id] = sum_inertia.inverse()

    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in range(self.ps.particle_num[None]):
            if not self.ps.is_static_rigid_body(p_i, self.step_num[None]):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(self.step_num[None], p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[self.step_num[None], p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        if self.ps.material[self.step_num[None], p_j] == self.ps.material_solid:
            delta += self.cubic_kernel((self.ps.x[self.step_num[None], p_i] - self.ps.x[self.step_num[None], p_j]).norm())


    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in range(self.ps.particle_num[None]):
            if not self.ps.is_dynamic_rigid_body(p_i, self.step_num[None]):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(self.step_num[None], p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[self.step_num[None], p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[self.step_num[None], self.iter_num[None], p_i] -= (
            1.0 + c_f) * self.ps.v[self.step_num[None], self.iter_num[None], p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type:int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[self.step_num[None], p_i] == particle_type and self.ps.is_dynamic[self.step_num[None], p_i]: 
                pos = self.ps.x[self.step_num[None], p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[self.step_num[None], p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[self.step_num[None], p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[self.step_num[None], p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[self.step_num[None], p_i][1] = self.ps.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[self.step_num[None], p_i] == particle_type and self.ps.is_dynamic[self.step_num[None], p_i]:
                pos = self.ps.x[self.step_num[None], p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[self.step_num[None], p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[self.step_num[None], p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[self.step_num[None], p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[self.step_num[None], p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[self.step_num[None], p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[self.step_num[None], p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)


    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.object_id[0, p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[0, p_i]
                cm += mass * self.ps.x[0, p_i]
                sum_m += mass
        cm /= sum_m
        return cm
    

    @ti.kernel
    def compute_com_kernel(self, object_id: int)->ti.types.vector(3, float):
        return self.compute_com(object_id)
        

    # @ti.kernel
    # def compute_rigid_collision(self):
    #     # FIXME: This is a workaround, rigid collision failure in some cases is expected
    #     for p_i in range(self.ps.particle_num[None]):
    #         if not self.ps.is_dynamic_rigid_body(p_i):
    #             continue
    #         cnt = 0
    #         x_delta = ti.Vector([0.0 for i in range(self.ps.dim)])
    #         for j in range(self.ps.solid_neighbors_num[p_i]):
    #             p_j = self.ps.solid_neighbors[p_i, j]

    #             if self.ps.is_static_rigid_body(p_i):
    #                 cnt += 1
    #                 x_j = self.ps.x[p_j]
    #                 r = self.ps.x[p_i] - x_j
    #                 if r.norm() < self.ps.particle_diameter:
    #                     x_delta += (r.norm() - self.ps.particle_diameter) * r.normalized()
    #         if cnt > 0:
    #             self.ps.x[p_i] += 2.0 * x_delta # / cnt
                        



    # def solve_rigid_body(self):
    #     for i in range(1):
    #         for r_obj_id in self.ps.object_id_rigid_body:
    #             if self.ps.object_collection[r_obj_id]["isDynamic"]:
    #                 R = self.solve_constraints(r_obj_id)

    #                 if self.ps.cfg.get_cfg("exportObj"):
    #                     # For output obj only: update the mesh
    #                     cm = self.compute_com_kernel(r_obj_id)
    #                     ret = R.to_numpy() @ (self.ps.object_collection[r_obj_id]["restPosition"] - self.ps.object_collection[r_obj_id]["restCenterOfMass"]).T
    #                     self.ps.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T

    #                 # self.compute_rigid_collision()
    #                 self.enforce_boundary_3D(self.ps.material_solid)


    @ti.kernel
    def solve_rigid_body(self):
        for r_obj_id in range(self.ps.num_objects):
            if self.ps.is_rigid[r_obj_id] == 1:
                self.ps.rigid_force[self.step_num[None], r_obj_id] += self.ps.rigid_mass[r_obj_id] * ti.Vector(self.g)

                self.ps.rigid_v[self.step_num[None] + 1, r_obj_id] = self.ps.rigid_v[self.step_num[None], r_obj_id] + self.dt[None] * self.ps.rigid_force[self.step_num[None], r_obj_id] / self.ps.rigid_mass[r_obj_id]
                self.ps.rigid_force[self.step_num[None] + 1, r_obj_id].fill(0.0)
                self.ps.rigid_x[self.step_num[None] + 1, r_obj_id] = self.ps.rigid_x[self.step_num[None], r_obj_id] + self.dt[None] * self.ps.rigid_v[self.step_num[None] + 1, r_obj_id]
                self.ps.rigid_omega[self.step_num[None] + 1, r_obj_id] = self.ps.rigid_omega[self.step_num[None], r_obj_id] + self.dt[None] * self.ps.rigid_inv_inertia[self.step_num[None], r_obj_id] @ self.ps.rigid_torque[self.step_num[None], r_obj_id]
                self.ps.rigid_torque[self.step_num[None], r_obj_id].fill(0.0)
                self.ps.rigid_quaternion[self.step_num[None], r_obj_id] += self.dt[None] * 0.5 * quaternion_multiply(vec32quaternion(self.ps.rigid_omega[self.step_num[None], r_obj_id]), self.ps.rigid_quaternion[self.step_num[None], r_obj_id])
                self.ps.rigid_quaternion[self.step_num[None] + 1, r_obj_id] = self.ps.rigid_quaternion[self.step_num[None], r_obj_id].normalized()
                R = quaternion2rotation_matrix(self.ps.rigid_quaternion[self.step_num[None] + 1, r_obj_id])
                self.ps.rigid_inertia[self.step_num[None] + 1, r_obj_id] = R @ self.ps.rigid_inertia0[r_obj_id] @ R.transpose()
                self.ps.rigid_inv_inertia[self.step_num[None] + 1, r_obj_id] = self.ps.rigid_inertia[self.step_num[None] + 1, r_obj_id].inverse()
            
    @ti.kernel
    def update_rigid_particle_info(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i, self.step_num[None]):
                r = self.ps.object_id[self.step_num[None], p_i]
                x_rel = self.ps.x_0[self.step_num[None], p_i] - self.ps.rigid_rest_cm[r]
                self.ps.x[self.step_num[None], p_i] = self.ps.rigid_x[self.step_num[None] + 1, r] + quaternion2rotation_matrix(self.ps.rigid_quaternion[self.step_num[None] + 1, r]) @ x_rel
                self.ps.v[self.step_num[None], 0, p_i] = self.ps.rigid_v[self.step_num[None] + 1, r] + self.ps.rigid_omega[self.step_num[None] + 1, r].cross(x_rel)


    def step(self, step):
        print(f"------------step {step}------------")
        last_iter = self.iter_num[None]
        self.step_num[None] = step
        self.iter_num[None] = 0
        self.ps.initialize_particle_system(step, last_iter)
        self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        self.update_rigid_particle_info()
        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        elif self.ps.dim == 3:
            self.enforce_boundary_3D(self.ps.material_fluid)
    

    def end(self):
        return self.step_num[None] >= self.ps.steps - 1
