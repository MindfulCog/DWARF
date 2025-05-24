import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def live_dashboard(sim):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("DWARF Tidal Simulator - Live Dashboard")

    # Setup for displacement plot
    displacement_line, = ax1.plot([], [], label="Avg Displacement")
    ax1.set_xlim(0, sim.total_steps)
    ax1.set_ylim(0, 100)  # will auto-adjust
    ax1.set_ylabel("Displacement (m)")
    ax1.set_title("Average Tidal Displacement")
    ax1.grid(True)

    # Setup for momentum plot
    momentum_line, = ax2.plot([], [], label="Angular Momentum (Z)")
    ax2.set_xlim(0, sim.total_steps)
    ax2.set_ylim(-1e10, 1e10)  # placeholder range
    ax2.set_ylabel("Angular Momentum")
    ax2.set_title("Angular Momentum Over Time")
    ax2.grid(True)

    xdata, ydata_disp, ydata_mom = [], [], []

    def update(frame):
        sim_step = frame
        sim.run_step(sim_step)

        avg_disp = np.mean(np.linalg.norm(sim.positions - sim.tracers, axis=0))
        Lz = np.sum(np.cross(sim.positions.T, sim.velocities.T)[:, 2])

        xdata.append(sim_step)
        ydata_disp.append(avg_disp)
        ydata_mom.append(Lz)

        displacement_line.set_data(xdata, ydata_disp)
        momentum_line.set_data(xdata, ydata_mom)

        ax1.set_ylim(0, max(ydata_disp) * 1.1)
        ax2.set_ylim(min(ydata_mom) * 1.1, max(ydata_mom) * 1.1)
        return displacement_line, momentum_line

    ani = FuncAnimation(fig, update, frames=sim.total_steps, interval=100, repeat=False)
    plt.tight_layout()
    plt.show()
