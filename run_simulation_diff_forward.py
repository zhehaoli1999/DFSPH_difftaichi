import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system_diff_forward import ParticleSystem
from DFSPH_diff_forward import DFSPHSolver

arch = ti.cpu
assert arch is ti.gpu or arch is ti.cpu

ti.init(arch=arch, device_memory_fraction=0.5)

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

    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_frames = config.get_cfg("exportFrame")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"{scene_name}_output", exist_ok=True)


    ps = ParticleSystem(config, GGUI=True, arch=arch, debug=True)

    solver = build_solver(ps)
    solver.initialize()

    cnt_frame = 0
    losses = []
    loss_grad = ti.Vector([0.0, 0.0, 0.0])
    MAX_OPT_NUM = 50

    opt_time = 0
    lr = 4.0

    while True:
        opt_time += 1
        with ti.ad.FwdMode(loss=ps.loss, param=ps.rigid_adjust_v, seed=[1.0,0.0,0.0]):
            solver.initialize_from_restart()
            while not solver.end(cnt_frame):
                solver.step(cnt_frame)
                print(cnt_frame)
                cnt_frame += 1
            solver.compute_loss(ps.steps - 1)
            cnt_frame = 0

        loss_grad[0] = ps.loss.dual[None]
        
        # with ti.ad.FwdMode(loss=ps.loss, param=ps.rigid_adjust_v, seed=[0.0,1.0,0.0]):
        #     solver.initialize_from_restart()
        #     while not solver.end(cnt_frame):
        #         solver.step(cnt_frame)
        #         cnt_frame += 1
        #     solver.compute_loss(ps.steps - 1)
        #     cnt_frame = 0

        # loss_grad[1] = ps.loss.dual[None]
        
        # with ti.ad.FwdMode(loss=ps.loss, param=ps.rigid_adjust_v, seed=[0.0,0.0,1.0]):
        #     solver.initialize_from_restart()
        #     while not solver.end(cnt_frame):
        #         solver.step(cnt_frame)
        #         cnt_frame += 1
        #     solver.compute_loss(ps.steps - 1)
        #     cnt_frame = 0
        
        # loss_grad[2] = ps.loss.dual[None]

        # current_loss = ps.loss[None]
        # losses.append(current_loss)
        # print("loss: ", current_loss)
        # print("loss grad: ", loss_grad)

        ps.print_rigid_grad_info(ps.steps-1, "rigid.log")
        exit()

        # if current_loss < 1e-3 or opt_time > MAX_OPT_NUM:
        #     print(losses)
        #     break
        # solver.update(lr, loss_grad)

    ps.close()
