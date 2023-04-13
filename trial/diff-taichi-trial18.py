import taichi as ti

'''
strange bug: field.dual have a dtype mistake in debug mode
'''

ti.init(arch=ti.cpu, debug=True)

@ti.data_oriented
class ParticleSystem:
    def __init__(self) -> None:
        self.dim = 3
        self.NUM_STEPS = 5
        self.NUM_OBJECTS = 2
        self.x = ti.Vector.field(self.dim, dtype=float, needs_dual=True)
        ti.root.dense(ti.ij, (self.NUM_STEPS, self.NUM_OBJECTS)).place(self.x, self.x.dual)
        print(self.x.dual.dtype, self.x.dtype)

ps = ParticleSystem()