# scripts/view_grid_and_path.py
import matplotlib.pyplot as plt
from sim.map_grid import make_demo_world
from sim.planners.dummy_planner import fakeplan_squiggly 
from sim.viz.draw import draw_grid, draw_path

def main():
    world = make_demo_world()
    path = fakeplan_squiggly(world)

    fig, ax = plt.subplots()
    draw_grid(
        ax,
        world.grid,
        world.start,
        world.goal,
        resolution=world.resolution,
        center_cell=world.center_cell,
    )
    draw_path(ax, path)
    plt.title("Grid + Custom Path")
    plt.show()

if __name__ == "__main__":
    main()
