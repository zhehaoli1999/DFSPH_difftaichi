import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system_diff_forward import ParticleSystem
from DFSPH_diff_forward import DFSPHSolver

arch = ti.gpu
assert arch is ti.gpu or arch is ti.cpu

ti.init(arch=arch, device_memory_fraction=0.9)

def build_solver(ps: ParticleSystem):
    solver_type = ps.cfg.get_cfg("simulationMethod")
    # if solver_type == 0:
    #     return WCSPHSolver(ps)
    # elif solver_type == 4:
    if solver_type == 4:
        return DFSPHSolver(ps)
    else:
        raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]

    ps = ParticleSystem(config, GGUI=True, arch=arch)

    solver = build_solver(ps)
    solver.initialize()

    cnt_frame = 0
    losses = []
    loss_grad = ti.Vector([0.0, 0.0, 0.0])
    MAX_OPT_NUM = 50

    opt_time = 0
    lr = 4.0

    window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    camera_position = ti.Vector([5.5, 2.5, 4.0])
    camera_move_step = 0.01
    camera.position(camera_position[0], camera_position[1], camera_position[2])
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, 0.0)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []

    solver.initialize_from_restart()

    paused = True

    while window.running:
        if not paused:
            print(cnt_frame)
            solver.step(cnt_frame)
            cnt_frame += 1
            ps.copy_to_vis_buffer(step=cnt_frame - 1, invisible_objects=invisible_objects)
            if solver.end(cnt_frame):
                break

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key == 'r':
                solver.initialize_from_restart()
                cnt_frame = 0
            elif e.key == 'p':
                print(camera_position)
        
        if window.is_pressed('w'):
            camera_position[0] += camera_move_step
        if window.is_pressed('s'):
            camera_position[0] -= camera_move_step
        if window.is_pressed('a'):
            camera_position[2] += camera_move_step
        if window.is_pressed('d'):
            camera_position[2] -= camera_move_step
        if window.is_pressed('z'):
            camera_position[1] += camera_move_step
        if window.is_pressed('x'):
            camera_position[1] -= camera_move_step
        
        camera.position(camera_position[0], camera_position[1], camera_position[2])
        camera.up(0.0, 1.0, 0.0)
        camera.lookat(-1.0, 0.0, 0.0)
        camera.fov(70)
        scene.set_camera(camera)

        scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)

        canvas.scene(scene)
        window.show()
    solver.compute_loss(ps.steps - 1)
    cnt_frame = 0
    # ps.print_rigid_grad_info(ps.steps - 1, "rigid_0.01.log")
    # ps.close()
    exit()

