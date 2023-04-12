import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system_diff import ParticleSystem
from DFSPH_diff import DFSPHSolver


ti.init(arch=ti.gpu, device_memory_fraction=0.5, debug=True)

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


    ps = ParticleSystem(config, GGUI=True)

    solver = build_solver(ps)
    solver.initialize()

    cnt_frame = 0
    losses = []

    while True:
        # TODO:
        # the code cannot work this way
        # some replaced function grads should be modified
        with ti.ad.Tape(loss=ps.loss, validation=True):
            solver.initialize_from_restart()
            while not solver.end(cnt_frame):
                print(cnt_frame)
                solver.step(cnt_frame)
                cnt_frame += 1
            print("finish")
            solver.compute_loss(ps.steps - 1)
        current_loss = ps.loss[None]
        losses.append(current_loss)
        print("loss: ", current_loss)
        solver.update()
