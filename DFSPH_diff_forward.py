import taichi as ti
from sph_base_diff_forward import SPHBase
from particle_system_diff_forward import ParticleSystem

def none(*args):
    pass
print_debug = none

class DFSPHSolver(SPHBase):
    def __init__(self, particle_system: ParticleSystem):
        super().__init__(particle_system)
        
        self.surface_tension = 0.01

        self.enable_divergence_solver = True

        self.m_max_iterations_v = self.ps.max_iter // 2
        self.m_max_iterations = self.ps.max_iter // 2

        self.m_eps = 1e-5

        self.max_error_V = 0.1
        self.max_error = 0.05
    

    @ti.func
    def compute_densities_task(self, step, iter, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[step, p_i]
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[step, p_j]
            ret += self.ps.m_V[step, p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[step, p_j]
            ret += self.ps.m_V[step, p_j] * self.cubic_kernel((x_i - x_j).norm())


    @ti.kernel
    def compute_densities(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                
                self.ps.density[step, p_i] = self.ps.m_V[step, p_i] * self.cubic_kernel(0.0)
                den = 0.0
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_densities_task, den)
                self.ps.density[step, p_i] += den
                self.ps.density[step, p_i] *= self.density_0
    

    @ti.func
    def compute_non_pressure_forces_task(self, step, iter, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[step, p_i]
        
        ############## Surface Tension ###############
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[step, p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[step, p_i] * self.ps.m[step, p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[step, p_i] * self.ps.m[step, p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
            
        
        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[step, p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[step, iter, p_i] -
                self.ps.v[step, iter, p_j]).dot(r)
        
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[step, p_j] / (self.ps.density[step, p_j])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[step, p_j] / (self.ps.density[step, p_i])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j, step):
                r_id = self.ps.object_id[step, p_j]
                force = -f_v * self.ps.density[step, p_i] * self.ps.m_V[step, p_i]
                self.ps.rigid_force[step, r_id] += force
                self.ps.rigid_torque[step, r_id] += (self.ps.x[step, p_j] - self.ps.rigid_x[step, r_id]).cross(force)


    @ti.kernel
    def compute_non_pressure_forces(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[step, p_i] = d_v
    

    @ti.kernel
    def advect(self, step: int, iter: int):
        # Update position
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[step, p_i] and self.ps.material[step, p_i] == self.ps.material_fluid:
                self.ps.x_buffer[step, p_i] = self.ps.x[step, p_i] + self.dt[None] * self.ps.v[step, iter, p_i]
    

    @ti.kernel
    def compute_DFSPH_factor(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                sum_grad_p_k = 0.0
                grad_p_i = ti.Vector([0.0 for _ in range(self.ps.dim)])
                
                # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
                ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
                
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_DFSPH_factor_task, ret)
                
                sum_grad_p_k = ret[3]
                for i in ti.static(range(3)):
                    grad_p_i[i] = ret[i]
                sum_grad_p_k += grad_p_i.norm_sqr()

                # Compute pressure stiffness denominator
                factor = 0.0
                if sum_grad_p_k > 1e-6:
                    factor = -1.0 / sum_grad_p_k
                else:
                    factor = 0.0
                self.ps.dfsph_factor[step, p_i] = factor
            

    @ti.func
    def compute_DFSPH_factor_task(self, step, iter, p_i, p_j, ret: ti.template()):
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            grad_p_j = -self.ps.m_V[step, p_j] * self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j])
            ret[3] += grad_p_j.norm_sqr() # sum_grad_p_k
            for i in ti.static(range(3)): # grad_p_i
                ret[i] -= grad_p_j[i]
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            grad_p_j = -self.ps.m_V[step, p_j] * self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j])
            for i in ti.static(range(3)): # grad_p_i
                ret[i] -= grad_p_j[i]
    

    @ti.kernel
    def compute_density_change(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                ret = ti.Struct(density_adv=0.0, num_neighbors=0)
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_density_change_task, ret)

                # only correct positive divergence
                density_adv = ti.max(ret.density_adv, 0.0)
                num_neighbors = ret.num_neighbors

                # Do not perform divergence solve when paritlce deficiency happens
                if self.ps.dim == 3:
                    if num_neighbors < 20:
                        density_adv = 0.0
                else:
                    if num_neighbors < 7:
                        density_adv = 0.0
        
                self.ps.density_adv[step, iter, p_i] = density_adv


    @ti.func
    def compute_density_change_task(self, step, iter, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[step, iter, p_i]
        v_j = self.ps.v[step, iter, p_j]
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret.density_adv += self.ps.m_V[step, p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j]))
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret.density_adv += self.ps.m_V[step, p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j]))
        
        # Compute the number of neighbors
        ret.num_neighbors += 1
    

    @ti.kernel
    def compute_density_adv(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                delta = 0.0
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_density_adv_task, delta)
                density_adv = self.ps.density[step, p_i] /self.density_0 + self.dt[None] * delta
                self.ps.density_adv[step, iter, p_i] = ti.max(density_adv, 1.0)


    @ti.func
    def compute_density_adv_task(self, step, iter, p_i, p_j, ret: ti.template()):
        v_i = self.ps.v[step, iter, p_i]
        v_j = self.ps.v[step, iter, p_j]
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            ret += self.ps.m_V[step, p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j]))
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            ret += self.ps.m_V[step, p_j] * (v_i - v_j).dot(self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j]))


    @ti.kernel
    def compute_density_error(self, step: int, iter: int, offset: float) -> float:
        density_error = 0.0
        for I in range(self.ps.particle_num[None]):
            if self.ps.material[step, I] == self.ps.material_fluid:
                density_error += self.density_0 * self.ps.density_adv[step, iter, I] - offset
        return density_error

    def divergence_solve(self):
        # TODO: warm start 
        # Compute velocity of density change
        self.compute_density_change(self.step_num, self.iter_num[self.step_num])

        m_iterations_v = 0
        
        # Start solver
        avg_density_err = 0.0

        while m_iterations_v < 1 or m_iterations_v < self.m_max_iterations_v:
            
            avg_density_err = self.divergence_solver_iteration()
            # Max allowed density fluctuation
            # use max density error divided by time step size
            eta = 1.0 / self.dt[None] * self.max_error_V * 0.01 * self.density_0
            # print("eta ", eta)
            
            # move it inside to make sure same counter behavior in different situations
            m_iterations_v += 1
            if avg_density_err <= eta:
                break
        # print(f"DFSPH - iteration V: {m_iterations_v} Avg density err: {avg_density_err}")

        # Multiply by h, the time step size has to be removed 
        # to make the stiffness value independent 
        # of the time step size
        
        self.divergence_iter_num[self.step_num] = m_iterations_v

        # TODO: if warm start
        # also remove for kappa v


    def divergence_solver_iteration(self):
        print_debug("kernel")
        self.divergence_solver_iteration_kernel(self.step_num, self.iter_num[self.step_num])
        self.iter_num[self.step_num] += 1
        print_debug("density change")
        self.compute_density_change(self.step_num, self.iter_num[self.step_num])
        print_debug("error")
        density_err = self.compute_density_error(self.step_num, self.iter_num[self.step_num], 0.0)
        return density_err / self.ps.fluid_particle_num


    @ti.kernel
    def divergence_solver_iteration_kernel(self, step: int, iter: int):
        # Perform Jacobi iteration
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                # evaluate rhs
                b_i = self.ps.density_adv[step, iter, p_i]
                k_i = b_i * self.ps.dfsph_factor[step, p_i] * self.inv_dt[None]
                ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.ps.dim)]), k_i=k_i)
                # TODO: if warm start
                # get_kappa_V += k_i
                self.ps.for_all_neighbors(step, iter, p_i, self.divergence_solver_iteration_task, ret)
                self.ps.v[step, iter + 1, p_i] = self.ps.v[step, iter, p_i] + ret.dv
        
    
    @ti.func
    def divergence_solver_iteration_task(self, step, iter, p_i, p_j, ret: ti.template()):
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[step, iter, p_j]
            k_j = b_j * self.ps.dfsph_factor[step, p_j] * self.inv_dt[None]
            k_sum = ret.k_i + self.density_0 / self.density_0 * k_j  # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.ps.m_V[step, p_j] * self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j])
                ret.dv -= self.dt[None] * k_sum * grad_p_j
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = -self.ps.m_V[step, p_j] * self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j])
                vel_change =  -self.dt[None] * 1.0 * ret.k_i * grad_p_j
                ret.dv += vel_change
                if self.ps.is_dynamic_rigid_body(p_j, step):
                    self.divergence_contact_num[None] += 1
                    r_id = self.ps.object_id[step, p_j]
                    force = -vel_change * (1 / self.dt[None]) * self.ps.density[step, p_i] * self.ps.m_V[step, p_i]
                    self.ps.rigid_force[step, r_id] += force
                    self.ps.rigid_torque[step, r_id] += (self.ps.x[step, p_j] - self.ps.rigid_x[step, r_id]).cross(force)


    def pressure_solve(self):

        # TODO: warm start
        
        # Compute rho_adv
        self.compute_density_adv(self.step_num, self.iter_num[self.step_num])

        m_iterations = 0

        # Start solver
        avg_density_err = 0.0

        while m_iterations < 1 or m_iterations < self.m_max_iterations:
            
            avg_density_err = self.pressure_solve_iteration()
            # Max allowed density fluctuation
            eta = self.max_error * 0.01 * self.density_0
            
            # move it inside to make sure same counter behavior in different situations
            m_iterations += 1
            if avg_density_err <= eta:
                break
        # print(f"DFSPH - iterations: {m_iterations} Avg density Err: {avg_density_err:.4f}")
        # Multiply by h, the time step size has to be removed 
        # to make the stiffness value independent 
        # of the time step size

        self.pressure_iter_num[self.step_num] = m_iterations

        # TODO: if warm start
        # also remove for kappa v

    
    def pressure_solve_iteration(self):
        print_debug("kernel")
        self.pressure_solve_iteration_kernel(self.step_num, self.iter_num[self.step_num])
        self.iter_num[self.step_num] += 1
        print_debug("density adv")
        self.compute_density_adv(self.step_num, self.iter_num[self.step_num])
        print_debug("error")
        density_err = self.compute_density_error(self.step_num, self.iter_num[self.step_num], self.density_0)
        return density_err / self.ps.fluid_particle_num

    
    @ti.kernel
    def pressure_solve_iteration_kernel(self, step: int, iter: int):
        # Compute pressure forces
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                # Evaluate rhs
                b_i = self.ps.density_adv[step, iter, p_i] - 1.0
                k_i = b_i * self.ps.dfsph_factor[step, p_i] * self.inv_dt2[None]
                ret = ti.Struct(dv=ti.Vector([0.0 for _ in range(self.ps.dim)]), k_i=k_i)

                # TODO: if warmstart
                # get kappa V
                self.ps.for_all_neighbors(step, iter, p_i, self.pressure_solve_iteration_task, ret)
                self.ps.v[step, iter + 1, p_i] = self.ps.v[step, iter, p_i] + ret.dv
    

    @ti.func
    def pressure_solve_iteration_task(self, step, iter, p_i, p_j, ret: ti.template()):
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            b_j = self.ps.density_adv[step, iter, p_j] - 1.0
            k_j = b_j * self.ps.dfsph_factor[step, p_j] * self.inv_dt2[None]
            k_sum = ret.k_i + self.density_0 / self.density_0 * k_j # TODO: make the neighbor density0 different for multiphase fluid
            if ti.abs(k_sum) > self.m_eps:
                grad_p_j = -self.ps.m_V[step, p_j] * self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j])
                # Directly update velocities instead of storing pressure accelerations
                ret.dv -= self.dt[None] * k_sum * grad_p_j  # ki, kj already contain inverse density
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            if ti.abs(ret.k_i) > self.m_eps:
                grad_p_j = -self.ps.m_V[step, p_j] * self.cubic_kernel_derivative(self.ps.x[step, p_i] - self.ps.x[step, p_j])

                # Directly update velocities instead of storing pressure accelerations
                vel_change = - self.dt[None] * 1.0 * ret.k_i * grad_p_j  # kj already contains inverse density
                ret.dv += vel_change
                if self.ps.is_dynamic_rigid_body(p_j, step):
                    self.pressure_contact_num[None] += 1
                    r_id = self.ps.object_id[step, p_j]
                    force = -vel_change * (1 / self.dt[None]) * self.ps.density[step, p_i] * self.ps.m_V[step, p_i]
                    self.ps.rigid_force[step, r_id] += force
                    self.ps.rigid_torque[step, r_id] += (self.ps.x[step, p_j] - self.ps.rigid_x[step, r_id]).cross(force)


    @ti.kernel
    def predict_velocity(self, step: int, iter: int):
        # compute new velocities only considering non-pressure forces
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[step, p_i] and self.ps.material[step, p_i] == self.ps.material_fluid:
                self.ps.v[step, iter + 1, p_i] = self.ps.v[step, iter, p_i] + self.dt[None] * self.ps.acceleration[step, p_i]

    def substep(self):
        self.pressure_contact_num[None] = 0
        self.divergence_contact_num[None] = 0

        print_debug("compute density")
        self.compute_densities(self.step_num, self.iter_num[self.step_num])
        print_debug("compute factor")
        self.compute_DFSPH_factor(self.step_num, self.iter_num[self.step_num])
        if self.enable_divergence_solver:
            print_debug("divergence solve")
            self.divergence_solve()
            self.ps.debug_print(f"after divergence solve (iter number {self.divergence_iter_num[self.step_num]} contact number {self.divergence_contact_num[None]})\n")
            self.ps.debug_print_rigid_force(self.step_num)
        print_debug("non pressure force")
        self.compute_non_pressure_forces(self.step_num, self.iter_num[self.step_num])
        self.ps.debug_print("after viscosity\n")
        self.ps.debug_print_rigid_force(self.step_num)
        print_debug("predict velocity")
        self.predict_velocity(self.step_num, self.iter_num[self.step_num])
        self.iter_num[self.step_num] += 1
        print_debug("pressure solve")
        self.pressure_solve()
        self.ps.debug_print(f"after pressure solve (iter number {self.pressure_iter_num[self.step_num]} contact number {self.pressure_contact_num[None]})\n")
        self.ps.debug_print_rigid_force(self.step_num)
        print_debug("advect")
        self.advect(self.step_num, self.iter_num[self.step_num])


