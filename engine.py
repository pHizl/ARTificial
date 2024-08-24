import random
import math
import matplotlib.pyplot as plt
import numpy as np
from curves import CurveSet  # Ensure you have this module or correct the import
from graphics import *

class CrystalEnvironment(dict):
    def __init__(self, curves=None, **kw):
        self.curves = curves
        self._init_defaults()
        self.update(**kw)
        self.set_factory_settings()

    def set_factory_settings(self):
        self.factory_settings = self.copy()

    def step(self, x):
        if self.curves is None:
            return
        for key in self.curves:
            self[key] = self.curves[key][x]

    @classmethod
    def build_env(cls, name, steps, min_gamma=0.45, max_gamma=0.85):
        curves = {
            "beta": (1.3, 2),
            "theta": (0.01, 0.04),
            "alpha": (0.02, 0.1),
            "kappa": (0.001, 0.01),
            "mu": (0.01, 0.1),
            "upsilon": (0.00001, 0.0001),
            "sigma": (0.00001, 0.000001),
        }
        cs = CurveSet(name, steps, curves)
        cs.run_graph()
        env = {key: cs[key][0] for key in curves}
        env["gamma"] = random.random() * (max_gamma - min_gamma) + min_gamma
        return CrystalEnvironment(curves=cs, **env)

    def _init_defaults(self):
        self["beta"] = 1.3
        self["theta"] = 0.025
        self["alpha"] = 0.08
        self["kappa"] = 0.003
        self["mu"] = 0.07
        self["upsilon"] = 0.00005
        self["sigma"] = 0.00001
        self["gamma"] = 0.5


class CrystalLattice:
    def __init__(self, size, environment=None, celltype=None, max_steps=0, margin=None):
        self.size = size
        self.environment = environment if environment else CrystalEnvironment()
        self.celltype = celltype if celltype else SnowflakeCell
        self.iteration = 1
        self.margin = margin if margin else 0.85
        self.max_steps = max_steps
        self._init_cells()

    def _init_cells(self):
        self.cells = [None] * (self.size * self.size)
        for x in range(self.size):
            for y in range(self.size):
                xy = (x, y)
                cell = self.celltype(xy, self)
                idx = self._cell_index(xy)
                self.cells[idx] = cell
        center_pt = self._cell_index((self.size // 2, self.size // 2))
        self.cells[center_pt].attach(1)

    def _cell_index(self, xy):
        (x, y) = xy
        return y * self.size + x  # Simplified and fixed index calculation

    def crop_snowflake(self, margin=None):
        def scale(val):
            X_SCALE_FACTOR = (1.0 / math.sqrt(3))
            return int(round(X_SCALE_FACTOR * val))
        if margin == None:
            margin = 15
        half = self.size / 2
        radius = scale(self.snowflake_radius())
        distance = min(radius + margin, half)
        half_s = scale(half)
        distance_s = scale(distance)
        box = (half_s - distance, half - distance, half_s + distance, half + distance)
        return box

    def get_neighbors(self, xy):
        (x, y) = xy
        nlist = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y), (x - 1, y - 1), (x + 1, y + 1)]
        nlist = map(self._cell_index, filter(self._xy_ok, nlist))
        res = tuple([self.cells[nidx] for nidx in nlist if self.cells[nidx] != None])
        return res
    
    def _xy_ok(self, xy):
        (x, y) = xy
        return (x >= 0 and x < self.size and y >= 0 and y < self.size)

    def grow(self):
        while True:
            self.step()
            if not self.headroom():
                break

    def step(self):
        for cell in self.cells:
            if cell is None or cell.attached:
                continue
            cell.step_one()
        for cell in self.cells:
            if cell is None or cell.attached:
                continue
            cell.step_two()
        for cell in self.cells:
            if cell is None or cell.attached:
                continue
            cell.step_three()
        self.iteration += 1
        self.environment.step(self.iteration)

    def headroom(self):
        if self.max_steps and self.iteration >= self.max_steps:
            return False
        cutoff = int(round(self.margin * (self.size / 2.0)))
        radius = self.snowflake_radius()
        if radius > cutoff:
            return False
        return True

    def snowflake_radius(self, angle=135):
        radius = 0
        half = self.size / 2.0
        while radius < half:
            radius += 1
            xy = self.polar_to_xy((angle, radius))
            cell = self.cells[self._cell_index(xy)]
            if cell.attached or cell.boundary:
                continue
            return radius
        return int(round(half))

    def polar_to_xy(self, args):
        (angle, distance) = args
        half = self.size / 2.0
        angle = math.radians(angle)
        y = int(round(half - (math.sin(angle) * distance)))
        x = int(round(half + (math.cos(angle) * distance)))
        return x, y

    def plot_snowflake(self):
        grid = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                cell = self.cells[self._cell_index((x, y))]
                if cell and cell.attached:
                    grid[y, x] = 1
        
        plt.imshow(grid, cmap='cool', interpolation='nearest')
        plt.title("Generated Snowflake")
        plt.axis('off')
        plt.savefig('snowflake.png')  # Save the plot to a file
        plt.close()  # Close the plot to free up memory
        
    def save_image(self, fn, **kw):
        r = RenderSnowflake(self)
        r.save_image(fn, **kw)

class SnowflakeCell:
    def __init__(self, xy, lattice):
        self.xy = xy
        self.lattice = lattice
        self.env = lattice.environment
        self.diffusive_mass = self.env["gamma"]
        self.boundary_mass = 0.0
        self.crystal_mass = 0.0
        self.attached = False
        self.boundary = 0
        self.attached_neighbors = []
        self.__neighbors = None
        self.age = 0  # Initialize age attribute

    @property
    def neighbors(self):
        if self.__neighbors is None:
            self.__neighbors = self.lattice.get_neighbors(self.xy)
        return self.__neighbors

    def update_boundary(self):
        self.boundary = not self.attached and any(cell.attached for cell in self.neighbors)

    def step_one(self):
        self.update_boundary()
        if self.boundary:
            self.attached_neighbors = [cell for cell in self.neighbors if cell.attached]
        self._next_dm = self.diffusion_calc()

    def step_two(self):
        self.diffusive_mass = self._next_dm
        self.attachment_flag = self.attached
        self.freezing_step()
        self.attachment_flag = self.attachment_step()
        self.melting_step()

    def step_three(self):
        if self.boundary and self.attachment_flag:
            self.attach()
        self.noise_step()

    def diffusion_calc(self):
        next_dm = self.diffusive_mass
        if self.attached:
            return next_dm
        self.age += 1
        for cell in self.neighbors:
            if cell.attached:
                next_dm += self.diffusive_mass
            else:
                next_dm += cell.diffusive_mass
        return next_dm / (len(self.neighbors) + 1)

    def attach(self, offset=0.0):
        self.crystal_mass = self.boundary_mass + self.crystal_mass + offset
        self.boundary_mass = 0
        self.attached = True

    def freezing_step(self):
        if not self.boundary:
            return
        self.boundary_mass += (1 - self.env["kappa"]) * self.diffusive_mass
        self.crystal_mass += (self.env["kappa"] * self.diffusive_mass)
        self.diffusive_mass = 0

    def attachment_step(self):
        if not self.boundary:
            return False
        attach_count = len(self.attached_neighbors)
        if attach_count <= 2:
            if self.boundary_mass > self.env["beta"]:
                return True
        elif attach_count == 3:
            if self.boundary_mass >= 1:
                return True
            else:
                summed_diffusion = self.diffusive_mass
                for cell in self.neighbors:
                    summed_diffusion += cell.diffusive_mass
                if summed_diffusion < self.env["theta"] and self.boundary_mass >= self.env["alpha"]:
                    return True
        elif attach_count >= 4:
            return True
        return False

    def melting_step(self):
        if not self.boundary:
            return
        self.diffusive_mass += self.env["mu"] * self.boundary_mass + self.env["upsilon"] * self.crystal_mass
        self.boundary_mass = (1 - self.env["mu"]) * self.boundary_mass
        self.crystal_mass = (1 - self.env["upsilon"]) * self.crystal_mass

    def noise_step(self):
        if self.boundary or self.attached:
            return
        if random.random() >= 0.5:
            self.diffusive_mass = (1 - self.env["sigma"]) * self.diffusive_mass
        else:
            self.diffusive_mass = (1 + self.env["sigma"]) * self.diffusive_mass

def run():
    size = 1000
    name = "Blake"
    max_steps = 500
    margin = 0.85

    environment = CrystalEnvironment.build_env(name, max_steps)
    cl = CrystalLattice(size, environment=environment, max_steps=max_steps, margin=margin)
    cl.grow()
    cl.save_image(f"{name}.png")  # Plot the snowflake using Matplotlib

if __name__ == "__main__":
    run()