import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig
from scan_single_buffer import parallel_prefix_sum_inclusive_inplace

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        # currently only 3-dim simulations are supported
        assert self.dim == 3
        # Simulation method
        self.simulation_method = self.cfg.get_cfg("simulationMethod")

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim

        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        self.grid_number = self.grid_num[0] * self.grid_num[1] * self.grid_num[2]
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        #========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        fluid_blocks = self.cfg.get_fluid_blocks()
        fluid_particle_num = 0
        for fluid in fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        #### Process Rigid Blocks ####
        rigid_blocks = self.cfg.get_rigid_blocks()
        rigid_particle_num = 0
        for rigid in rigid_blocks:
            particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num
        
        #### Process Rigid Bodies ####
        rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += voxelized_points_np.shape[0]
        
        self.fluid_particle_num = fluid_particle_num
        self.solid_particle_num = rigid_particle_num
        self.particle_max_num = fluid_particle_num + rigid_particle_num
        self.num_rigid_bodies = len(rigid_blocks)+len(rigid_bodies)

        self.num_objects = self.num_rigid_bodies + len(fluid_blocks)
        if len(rigid_blocks) > 0:
            print("Warning: currently rigid block functions are not completed, may lead to unexpected behaviour")
            input("Press Enter to continue")

        #### TODO: Handle the Particle Emitter ####
        # self.particle_max_num += emitted particles
        print(f"Current particle num: {self.particle_num[None]}, Particle max num: {self.particle_max_num}")

        self.steps = self.cfg.get_cfg("stepNum")
        self.max_iter = self.cfg.get_cfg("maxIterNum")

        #========== Allocate memory ==========#
        # Rigid body properties
        if self.num_rigid_bodies > 0:
            # TODO: Here we actually only need to store rigid boides, however the object id of rigid may not start from 0, so allocate center of mass for all objects
            self.rigid_rest_cm = ti.Vector.field(self.dim, dtype=float)
            self.rigid_x = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
            self.rigid_v0 = ti.Vector.field(self.dim, dtype=float)
            self.rigid_v = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
            self.rigid_quaternion = ti.Vector.field(4, dtype=float, needs_grad=True)
            self.rigid_omega = ti.Vector.field(3, dtype=float, needs_grad=True)
            self.rigid_omega0 = ti.Vector.field(3, dtype=float)
            self.rigid_force = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
            self.rigid_torque = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
            self.rigid_mass = ti.field(dtype=float)
            self.rigid_inertia = ti.Matrix.field(m=3, n=3, dtype=float, needs_grad=True)
            self.rigid_inertia0 = ti.Matrix.field(m=3, n=3, dtype=float)
            self.rigid_inv_mass = ti.field(dtype=float)
            self.rigid_inv_inertia = ti.Matrix.field(m=3, n=3, dtype=float, needs_grad=True)
            self.is_rigid = ti.field(dtype=int)
            ti.root.dense(ti.ij, (self.steps, self.num_objects)).place(self.rigid_x, self.rigid_v, self.rigid_quaternion, self.rigid_omega, self.rigid_force, self.rigid_torque,
                                                                                       self.rigid_inertia, self.rigid_inv_inertia)
            ti.root.dense(ti.ij, (self.steps, self.num_objects)).place(self.rigid_x.grad, self.rigid_v.grad, self.rigid_quaternion.grad, self.rigid_omega.grad,
                                                                        self.rigid_force.grad, self.rigid_torque.grad, self.rigid_inertia.grad, self.rigid_inv_inertia.grad)
            ti.root.dense(ti.i, (self.num_objects)).place(self.rigid_rest_cm, self.rigid_v0, self.rigid_omega0, self.rigid_mass, self.rigid_inertia0, self.rigid_inv_mass, self.is_rigid)

            self.rigid_adjust_x = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
            self.rigid_adjust_v = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
            self.rigid_adjust_omega = ti.Vector.field(3, dtype=float, needs_grad=True)
            self.rigid_adjust_quaternion = ti.Vector.field(4, dtype=float, needs_grad=True)

            ti.root.dense(ti.i, (self.num_objects)).place(self.rigid_adjust_x, self.rigid_adjust_v, self.rigid_adjust_omega, self.rigid_adjust_quaternion)
            ti.root.dense(ti.i, (self.num_objects)).place(self.rigid_adjust_x.grad, self.rigid_adjust_v.grad, self.rigid_adjust_omega.grad, self.rigid_adjust_quaternion.grad)

            for I in range(self.num_objects):
                self.rigid_adjust_quaternion[I] = ti.Vector([1., 0., 0., 0.])
                self.rigid_adjust_x[I] = ti.Vector([0., 0., 0.])
                self.rigid_adjust_v[I] = ti.Vector([0., 0., 0.])
                self.rigid_adjust_omega[I] = ti.Vector([0., 0., 0.])
        
        else:
            print("Error: rigid bodies must exist")
            exit()


        # Particle num of each grid
        self.grid_particles_num = ti.field(int, shape=int(self.grid_number))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_number))

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        # Particle related properties
        self.object_id = ti.field(dtype=int)
        self.x = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
        self.v = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
        self.m_V = ti.field(dtype=float, needs_grad=True)
        self.m = ti.field(dtype=float)
        self.density = ti.field(dtype=float, needs_grad=True)
        self.material = ti.field(dtype=int)
        self.color = ti.Vector.field(3, dtype=int)
        self.is_dynamic = ti.field(dtype=int)

        ti.root.dense(ti.ijk, (self.steps, self.max_iter, self.particle_max_num)).place(self.v, self.v.grad)
        ti.root.dense(ti.ij, (self.steps, self.particle_max_num)).place(self.object_id, self.x, self.x_buffer, self.x_0, self.acceleration, self.m_V, self.density, self.m, self.material, self.color, self.is_dynamic)
        ti.root.dense(ti.ij, (self.steps, self.particle_max_num)).place(self.x.grad, self.x_buffer.grad, self.x_0.grad, self.acceleration.grad,
                                                                        self.m_V.grad, self.density.grad)

        
        # used as "step -1" to satisfy resort operations
        self.init_temp_x = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
        self.init_temp_v = ti.Vector.field(self.dim, dtype=float, needs_grad=True)
        ti.root.dense(ti.i, (self.particle_max_num)).place(self.init_temp_x, self.init_temp_v, self.init_temp_x.grad, self.init_temp_v.grad)

        
        self.input_object_id = ti.field(dtype=int)
        self.input_x = ti.Vector.field(self.dim, dtype=float)
        self.input_v = ti.Vector.field(self.dim, dtype=float)
        self.input_m = ti.field(dtype=float)
        self.input_m_V = ti.field(dtype=float)
        self.input_density = ti.field(dtype=float)
        self.input_material = ti.field(dtype=int)
        self.input_color = ti.Vector.field(3, dtype=int)
        self.input_is_dynamic = ti.field(dtype=int)
        self.input_grid_ids = ti.field(dtype=int)
        self.input_grid_ids_new = ti.field(dtype=int)
        ti.root.dense(ti.i, (self.particle_max_num)).place(self.input_object_id, self.input_x, self.input_v, self.input_m, self.input_m_V, self.input_density, self.input_material, self.input_color, self.input_is_dynamic, self.input_grid_ids, self.input_grid_ids_new)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor = ti.field(dtype=float, needs_grad=True)
            self.density_adv = ti.field(dtype=float, needs_grad=True)
            ti.root.dense(ti.ij, (self.steps, self.particle_max_num)).place(self.dfsph_factor, self.dfsph_factor.grad)
            ti.root.dense(ti.ijk, (self.steps, self.max_iter, self.particle_max_num)).place(self.density_adv, self.density_adv.grad)

        self.loss = ti.field(dtype=float, shape=(), needs_grad=True)

        # Grid id for each particle
        self.grid_ids = ti.field(int)
        self.grid_ids_new = ti.field(int)
        ti.root.dense(ti.ij, (self.steps, self.particle_max_num)).place(self.grid_ids, self.grid_ids_new)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)


        #========== Initialize particles ==========#

        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.is_rigid[obj_id] = 0
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=1, # enforce fluid dynamic
                          color=color,
                          material=1) # 1 indicates fluid

        # Rigid bodies
        for rigid_body in rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
                if "angularVelocity" in rigid_body:
                    angular_velocity = np.array(rigid_body["angularVelocity"], dtype=np.float32)
                else:
                    angular_velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
                angular_velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body["color"], dtype=np.int32)
            self.rigid_v0[obj_id] = velocity
            self.rigid_omega0[obj_id] = angular_velocity
            self.is_rigid[obj_id] = 1
            print(obj_id, self.is_rigid[obj_id])
            self.rigid_rest_cm[obj_id] = rigid_body["restCenterOfMass"]
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.array([0 for _ in range(num_particles_obj)], dtype=np.int32), # material is solid
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32), # is_dynamic
                               np.stack([color for _ in range(num_particles_obj)])) # color


    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.input_object_id[p] = obj_id
        self.input_x[p] = x
        self.input_v[p] = v
        self.input_m[p] = self.m_V0 * density
        self.input_m_V[p] = self.m_V0
        self.input_density[p] = density
        self.input_material[p] = material
        self.input_is_dynamic[p] = is_dynamic
        self.input_color[p] = color
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):
        
        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      )

    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num


    @ti.func
    def pos_to_index(self, pos):
        return (pos / self.grid_size).cast(int)


    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
    
    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))
    

    @ti.func
    def is_static_rigid_body(self, p, step):
        return self.material[step, p] == self.material_solid and (not self.is_dynamic[step, p])


    @ti.func
    def is_dynamic_rigid_body(self, p, step):
        return self.material[step, p] == self.material_solid and self.is_dynamic[step, p]
    

    def print_grid_particles_num(self):
        for I in range(20000):
            print(I, self.grid_particles_num[I])


    @ti.kernel
    def update_grid_id(self, step: int):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in range(self.particle_num[None]):
            grid_index = 0
            if step != 0:
                grid_index = self.get_flatten_grid_index(self.x_buffer[step - 1, I])
            else:
                grid_index = self.get_flatten_grid_index(self.init_temp_x[I])
            if grid_index < 0:
                grid_index = 0
            elif grid_index >= self.grid_number:
                grid_index = self.grid_number - 1
            if step != 0:
                self.grid_ids[step - 1, I] = grid_index
            else:
                self.input_grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]
    
    @ti.kernel
    def counting_sort(self, step: int, last_iter: int):
        if step == 0:
            for i in range(self.particle_max_num):
                I = self.particle_max_num - 1 - i
                base_offset = 0
                if self.input_grid_ids[I] - 1 >= 0:
                    base_offset = self.grid_particles_num[self.input_grid_ids[I]-1]
                self.input_grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.input_grid_ids[I]], 1) - 1 + base_offset
            for I in range(self.particle_max_num):
                new_index = self.input_grid_ids_new[I]
                self.grid_ids[0, new_index] = self.input_grid_ids[I]
                self.object_id[0, new_index] = self.input_object_id[I]
                self.x_0[0, new_index] = self.init_temp_x[I]
                self.x[0, new_index] = self.init_temp_x[I]
                self.v[0, 0, new_index] = self.init_temp_v[I]
                self.m[0, new_index] = self.input_m[I]
                self.m_V[0, new_index] = self.input_m_V[I]
                self.density[0, new_index] = self.input_density[I]
                self.material[0, new_index] = self.input_material[I]
                self.color[0, new_index] = self.input_color[I]
                self.is_dynamic[0, new_index] = self.input_is_dynamic[I]
        else:
            for i in range(self.particle_max_num):
                I = self.particle_max_num - 1 - i
                base_offset = 0
                if self.grid_ids[step - 1, I] - 1 >= 0:
                    base_offset = self.grid_particles_num[self.grid_ids[step - 1, I]-1]
                self.grid_ids_new[step - 1, I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[step - 1, I]], 1) - 1 + base_offset
            for I in range(self.particle_max_num):
                new_index = self.grid_ids_new[step - 1, I]
                self.grid_ids[step, new_index] = self.grid_ids[step - 1, I]
                self.object_id[step, new_index] = self.object_id[step - 1, I]
                self.x_0[step, new_index] = self.x_0[step - 1, I]
                self.x[step, new_index] = self.x_buffer[step - 1, I]
                self.v[step, 0, new_index] = self.v[step - 1, last_iter, I]
                self.m_V[step, new_index] = self.m_V[step - 1, I]
                self.m[step, new_index] = self.m[step - 1, I]
                self.density[step, new_index] = self.density[step - 1, I]
                self.material[step, new_index] = self.material[step - 1, I]
                self.color[step, new_index] = self.color[step - 1, I]
                self.is_dynamic[step, new_index] = self.is_dynamic[step - 1, I]

                # if ti.static(self.simulation_method == 4):
                #     self.dfsph_factor[step, new_index] = self.dfsph_factor[step - 1, I]
                #     self.density_adv[step, new_index] = self.density_adv[step - 1, I]
    

    def initialize_particle_system(self, step, last_iter):
        self.update_grid_id(step)
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort(step, last_iter)
    

    @ti.func
    def for_all_neighbors(self, step, iter, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.x[step, p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            if grid_index < self.grid_number and grid_index >= 0:
                for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                    if p_i != p_j and (self.x[step, p_i] - self.x[step, p_j]).norm() < self.support_radius:
                        task(step, iter, p_i, p_j, ret)

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, step, invisible_objects=[]):
        if len(invisible_objects) != 0:
            self.x_vis_buffer.fill(0.0)
            self.color_vis_buffer.fill(0.0)
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(step, obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, step: int, obj_id: int):
        assert self.GGUI
        # FIXME: make it equal to actual particle num
        for i in range(self.particle_max_num):
            if self.object_id[step, i] == obj_id:
                self.x_vis_buffer[i] = self.x[step, i]
                self.color_vis_buffer[i] = self.color[step, i] / 255.0

    def dump(self, obj_id):
        np_object_id = self.object_id.to_numpy()
        mask = (np_object_id == obj_id).nonzero()
        np_x = self.x.to_numpy()[mask]
        np_v = self.v.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }
    

    def load_rigid_body(self, rigid_body):
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        rigid_body["mesh"] = mesh_backup
        is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        # voxelized_mesh.show()
        voxelized_points_np = voxelized_mesh.points
        print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")
        rigid_body["restPosition"] = voxelized_points_np
        rigid_body["restCenterOfMass"] = voxelized_points_np.mean(axis=0)
        
        return voxelized_points_np


    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.particle_diameter))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 is_dynamic,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        print('particle num ', num_new_particles)

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)


    # add for debug
    def print_rigid_info(self, step):
        for r in self.object_id_rigid_body:
            print("object ", r)
            print("x", self.rigid_x[step, r])
            print("x0", self.rigid_rest_cm[r])
            print("v", self.rigid_v[step, r])
            print("v0", self.rigid_v0[r])
            print("w", self.rigid_omega[step, r])
            print("w0", self.rigid_omega0[r])
            print("q", self.rigid_quaternion[step, r])

            print("m", self.rigid_mass[r])
            print("I", self.rigid_inertia[step, r])
            print("f", self.rigid_force[step, r])
            print("t", self.rigid_torque[step, r])