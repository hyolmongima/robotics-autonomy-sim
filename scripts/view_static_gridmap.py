# scripts/view_static_gridmap.py
import matplotlib.pyplot as plt
from sim.map_grid import make_demo_world
from sim.viz.draw import draw_grid

def main():
    world = make_demo_world()
    fig, ax = plt.subplots()
    draw_grid(ax, world.grid, world.start, world.goal)
    plt.title("Grid World")
    plt.show()

if __name__ == "__main__":
    main()