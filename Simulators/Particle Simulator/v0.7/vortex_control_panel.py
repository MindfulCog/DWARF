import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, CheckButtons
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import os

class VortexControlPanel:
    """
    Control panel for the DWARF Vortex Simulator.
    Provides a UI with sliders, buttons and visualizations to control the simulation.
    """
    
    def __init__(self, simulator, fig=None, initial_params=None):
        """
        Initialize the control panel.
        
        Args:
            simulator: DwarfVortexSimulator instance to control
            fig: Optional existing matplotlib figure
            initial_params: Optional dict of initial parameter values
        """
        self.simulator = simulator
        self.particles = simulator.particles
        
        # Setup figure and layout
        if fig is None:
            self.fig = plt.figure(figsize=(12, 9))
        else:
            self.fig = fig
        
        # Initial parameters
        self.params = {
            'memory_decay': 0.995,
            'global_drag': 0.001,
            'saturation_limit': 5.0,
            'show_field': True
        }
        
        # Update with provided parameters if any
        if initial_params:
            self.params.update(initial_params)
            
        # Apply initial parameters to simulator
        self.simulator.memory_decay = self.params['memory_decay']
        self.simulator.global_drag = self.params['global_drag']
        self.simulator.saturation_limit = self.params['saturation_limit']
        self.simulator.show_field = self.params['show_field']
        
        # Setup layout using GridSpec for responsive design
        self._setup_layout()
        
        # Create UI controls
        self._create_global_controls()
        self._create_particle_controls()
        self._create_view_controls()
        
        # Register update callbacks
        self._register_callbacks()
        
        # Adjust layout for better appearance
        self.fig.tight_layout()
    
    def _setup_layout(self) -> None:
        """Setup the layout grid for all UI elements."""
        # Main grid: simulation view at top, controls at bottom
        self.grid = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=self.fig)
        
        # Simulation view area
        self.ax_sim = self.fig.add_subplot(self.grid[0])
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_title('DWARF Vortex Simulator', fontsize=16)
        
        # Controls area subgrid
        self.controls_grid = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=self.grid[1], height_ratios=[1, 1, 0.5])
        
        # Global controls
        self.ax_controls1 = self.fig.add_subplot(self.controls_grid[0])
        self.ax_controls1.set_title('Global Parameters')
        self.ax_controls1.axis('off')
        
        # Particle controls
        self.ax_controls2 = self.fig.add_subplot(self.controls_grid[1])
        self.ax_controls2.set_title('Particle Controls')
        self.ax_controls2.axis('off')
        
        # View controls
        self.ax_controls3 = self.fig.add_subplot(self.controls_grid[2])
        self.ax_controls3.set_title('View Controls')
        self.ax_controls3.axis('off')
    
    def _create_global_controls(self) -> None:
        """Create sliders for global simulation parameters."""
        # Get control area bounds
        bbox = self.ax_controls1.get_position()
        x0, y0, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        
        # Calculate slider positions relative to the control area
        ctrl_width = 0.65 * width
        ctrl_height = 0.15 * height
        ctrl_left = x0 + 0.25 * width
        
        # Memory decay slider
        ax_memory = plt.axes([ctrl_left, y0 + 0.7 * height, ctrl_width, ctrl_height], facecolor='lightgoldenrodyellow')
        self.memory_slider = Slider(
            ax=ax_memory, 
            label='Memory Decay',
            valmin=0.95,
            valmax=0.999,
            valinit=self.params['memory_decay']
        )
        
        # Global drag slider
        ax_drag = plt.axes([ctrl_left, y0 + 0.4 * height, ctrl_width, ctrl_height], facecolor='lightgoldenrodyellow')
        self.drag_slider = Slider(
            ax=ax_drag, 
            label='Global Drag',
            valmin=0.0,
            valmax=0.01,
            valinit=self.params['global_drag']
        )
        
        # Saturation limit slider
        ax_sat = plt.axes([ctrl_left, y0 + 0.1 * height, ctrl_width, ctrl_height], facecolor='lightgoldenrodyellow')
        self.sat_slider = Slider(
            ax=ax_sat, 
            label='Saturation Limit',
            valmin=1.0,
            valmax=20.0,
            valinit=self.params['saturation_limit']
        )
    
    def _create_particle_controls(self) -> None:
        """Create sliders for individual particle parameters."""
        # Get control area bounds
        bbox = self.ax_controls2.get_position()
        x0, y0, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        
        # Check if there are particles to control
        if not self.particles:
            # Display message if no particles
            self.ax_controls2.text(0.5, 0.5, 'No particles to control', 
                                 horizontalalignment='center', 
                                 verticalalignment='center',
                                 transform=self.ax_controls2.transAxes)
            return
        
        # Setup control sizes and positioning
        num_particles = len(self.particles)
        slider_height = min(0.15 * height, 0.7 * height / num_particles) if num_particles > 0 else 0.15 * height
        ctrl_width = 0.65 * width
        ctrl_left = x0 + 0.25 * width
        
        # Create sliders for each particle's spin
        self.spin_sliders = []
        
        for i, particle in enumerate(self.particles):
            # Position vertically from top to bottom
            current_y = y0 + height * (0.85 - i * slider_height * 1.2 / height)
            
            # Create axis for slider
            ax_slider = plt.axes(
                [ctrl_left, current_y - slider_height/2, ctrl_width, slider_height], 
                facecolor='lightgoldenrodyellow'
            )
            
            # Create slider
            spin_slider = Slider(
                ax=ax_slider,
                label=f'{particle.name} Spin',
                valmin=-1.0,
                valmax=1.0,
                valinit=particle.spin,
                color=particle.color
            )
            
            self.spin_sliders.append(spin_slider)
    
    def _create_view_controls(self) -> None:
        """Create buttons and checkboxes for view control."""
        # Get control area bounds
        bbox = self.ax_controls3.get_position()
        x0, y0, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        
        # Reset button
        ax_reset = plt.axes([x0 + width * 0.1, y0 + height * 0.3, width * 0.15, height * 0.4])
        self.reset_button = Button(ax_reset, 'Reset')
        
        # Field toggle checkbox
        ax_toggle = plt.axes([x0 + width * 0.35, y0 + height * 0.3, width * 0.15, height * 0.4])
        self.field_toggle = CheckButtons(
            ax_toggle, 
            ['Show Field'], 
            [self.params['show_field']]
        )
        
        # Energy plot button
        ax_energy = plt.axes([x0 + width * 0.55, y0 + height * 0.3, width * 0.15, height * 0.4])
        self.energy_button = Button(ax_energy, 'Plot Energy')
        
        # Zoom buttons
        ax_zoom_in = plt.axes([x0 + width * 0.75, y0 + height * 0.5, width * 0.1, height * 0.3])
        self.zoom_in_button = Button(ax_zoom_in, '+')
        
        ax_zoom_out = plt.axes([x0 + width * 0.85, y0 + height * 0.5, width * 0.1, height * 0.3])
        self.zoom_out_button = Button(ax_zoom_out, '-')
        
        # Save button
        ax_save = plt.axes([x0 + width * 0.75, y0 + height * 0.1, width * 0.2, height * 0.3])
        self.save_button = Button(ax_save, 'Save Image')
    
    def _register_callbacks(self) -> None:
        """Register callback functions for UI elements."""
        # Global parameter sliders
        self.memory_slider.on_changed(self._update_memory_decay)
        self.drag_slider.on_changed(self._update_global_drag)
        self.sat_slider.on_changed(self._update_saturation_limit)
        
        # Particle spin sliders
        for i, slider in enumerate(self.spin_sliders):
            slider.on_changed(lambda val, idx=i: self._update_particle_spin(idx, val))
        
        # Buttons
        self.reset_button.on_clicked(self._reset_simulation)
        self.field_toggle.on_clicked(self._toggle_field)
        self.zoom_in_button.on_clicked(self._zoom_in)
        self.zoom_out_button.on_clicked(self._zoom_out)
        self.energy_button.on_clicked(self._show_energy_plot)
        self.save_button.on_clicked(self._save_screenshot)
        
        # Handle window resize
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
    
    def _update_memory_decay(self, value: float) -> None:
        """Update memory decay parameter."""
        self.simulator.set_memory_decay(value)
    
    def _update_global_drag(self, value: float) -> None:
        """Update global drag parameter."""
        self.simulator.set_global_drag(value)
    
    def _update_saturation_limit(self, value: float) -> None:
        """Update saturation limit parameter."""
        self.simulator.set_saturation_limit(value)
    
    def _update_particle_spin(self, particle_idx: int, value: float) -> None:
        """Update spin for a specific particle."""
        self.simulator.set_particle_spin(particle_idx, value)
    
    def _reset_simulation(self, event) -> None:
        """Reset the simulation."""
        self.simulator.reset()
    
    def _toggle_field(self, label) -> None:
        """Toggle field visibility."""
        self.simulator.toggle_field_visibility()
    
    def _zoom_in(self, event) -> None:
        """Zoom in on the simulation view."""
        xlim = self.ax_sim.get_xlim()
        ylim = self.ax_sim.get_ylim()
        
        # Zoom by 10%
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        new_x_range = x_range * 0.9
        new_y_range = y_range * 0.9
        
        self.ax_sim.set_xlim(x_center - new_x_range/2, x_center + new_x_range/2)
        self.ax_sim.set_ylim(y_center - new_y_range/2, y_center + new_y_range/2)
        self.fig.canvas.draw_idle()
    
    def _zoom_out(self, event) -> None:
        """Zoom out on the simulation view."""
        xlim = self.ax_sim.get_xlim()
        ylim = self.ax_sim.get_ylim()
        
        # Zoom out by 10%
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        new_x_range = x_range * 1.1
        new_y_range = y_range * 1.1
        
        self.ax_sim.set_xlim(x_center - new_x_range/2, x_center + new_x_range/2)
        self.ax_sim.set_ylim(y_center - new_y_range/2, y_center + new_y_range/2)
        self.fig.canvas.draw_idle()
    
    def _show_energy_plot(self, event) -> None:
        """Show energy conservation plot."""
        self.simulator.plot_energy()
    
    def _save_screenshot(self, event) -> None:
        """Save a screenshot of the current simulation state."""
        # Create output directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)
        
        # Save the figure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/dwarf_sim_{timestamp}.png"
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Screenshot saved as {filename}")
    
    def _on_resize(self, event) -> None:
        """Handle window resize events to maintain layout."""
        # Adjust layout for new window size
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
    
    def show(self) -> None:
        """Show the control panel."""
        plt.show()