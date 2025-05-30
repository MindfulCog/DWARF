import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
import os
from pathlib import Path
import webbrowser

class ParticleSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Simulator Controller")
        self.root.geometry("600x550")
        self.root.resizable(True, True)
        
        # Set minimum window size
        self.root.minsize(600, 550)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern theme
        
        # Configure colors
        bg_color = "#f0f0f0"
        frame_bg = "#ffffff"
        button_bg = "#4a86e8"
        button_fg = "#ffffff"
        
        self.root.configure(bg=bg_color)
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and description
        title_label = ttk.Label(main_frame, text="N-Body Particle Simulator", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        description = ("Configure simulation parameters below and click 'Run Simulation' to start.\n"
                      "The simulator will open in a new window.")
        desc_label = ttk.Label(main_frame, text=description, 
                             font=('Arial', 10))
        desc_label.pack(pady=5)
        
        # Create settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Simulation Settings", padding="10")
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create two columns
        left_frame = ttk.Frame(settings_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.Frame(settings_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Scenario selection
        ttk.Label(left_frame, text="Scenario:").pack(anchor=tk.W, pady=5)
        self.scenario_var = tk.StringVar(value="orbit")
        scenario_options = ["orbit", "random", "collision", "galaxy"]
        scenario_dropdown = ttk.Combobox(left_frame, textvariable=self.scenario_var, 
                                       values=scenario_options, state="readonly", width=15)
        scenario_dropdown.pack(fill=tk.X, pady=2)
        
        # Number of particles
        ttk.Label(left_frame, text="Number of Particles:").pack(anchor=tk.W, pady=5)
        self.particles_var = tk.IntVar(value=20)
        particles_spinbox = ttk.Spinbox(left_frame, from_=2, to=500, textvariable=self.particles_var, width=15)
        particles_spinbox.pack(fill=tk.X, pady=2)
        
        # Time step
        ttk.Label(left_frame, text="Time Step (smaller = more accurate):").pack(anchor=tk.W, pady=5)
        self.timestep_var = tk.DoubleVar(value=0.1)
        timestep_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        timestep_dropdown = ttk.Combobox(left_frame, textvariable=self.timestep_var, 
                                       values=timestep_values, width=15)
        timestep_dropdown.pack(fill=tk.X, pady=2)
        
        # Record frames option
        self.record_var = tk.BooleanVar(value=False)
        record_check = ttk.Checkbutton(right_frame, text="Record Frames", variable=self.record_var)
        record_check.pack(anchor=tk.W, pady=5)
        
        # Output directory
        ttk.Label(right_frame, text="Output Directory:").pack(anchor=tk.W, pady=5)
        
        output_frame = ttk.Frame(right_frame)
        output_frame.pack(fill=tk.X, pady=2)
        
        self.output_var = tk.StringVar(value="output")
        output_entry = ttk.Entry(output_frame, textvariable=self.output_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_button = ttk.Button(output_frame, text="Browse", command=self.browse_output)
        browse_button.pack(side=tk.RIGHT, padx=5)
        
        # Multiprocessing option
        self.multiprocessing_var = tk.BooleanVar(value=False)
        multiprocessing_check = ttk.Checkbutton(right_frame, text="Use Multiprocessing", 
                                              variable=self.multiprocessing_var)
        multiprocessing_check.pack(anchor=tk.W, pady=5)
        
        # Energy display option
        self.energy_var = tk.BooleanVar(value=False)
        energy_check = ttk.Checkbutton(right_frame, text="Show Energy Conservation", 
                                     variable=self.energy_var)
        energy_check.pack(anchor=tk.W, pady=5)
        
        # Advanced settings section
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Settings", padding="10")
        advanced_frame.pack(fill=tk.X, pady=10)
        
        # Screen resolution
        res_frame = ttk.Frame(advanced_frame)
        res_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(res_frame, text="Screen Resolution:").pack(side=tk.LEFT, padx=5)
        
        self.width_var = tk.IntVar(value=800)
        self.height_var = tk.IntVar(value=600)
        
        width_entry = ttk.Spinbox(res_frame, from_=640, to=1920, textvariable=self.width_var, width=5)
        width_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(res_frame, text="×").pack(side=tk.LEFT)
        
        height_entry = ttk.Spinbox(res_frame, from_=480, to=1080, textvariable=self.height_var, width=5)
        height_entry.pack(side=tk.LEFT, padx=5)
        
        # FPS limit
        fps_frame = ttk.Frame(advanced_frame)
        fps_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(fps_frame, text="FPS Limit:").pack(side=tk.LEFT, padx=5)
        
        self.fps_var = tk.IntVar(value=60)
        fps_options = [30, 60, 90, 120, 144, 240]
        fps_dropdown = ttk.Combobox(fps_frame, textvariable=self.fps_var,
                                   values=fps_options, width=5, state="readonly")
        fps_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Command preview
        preview_frame = ttk.LabelFrame(main_frame, text="Command Preview", padding="10")
        preview_frame.pack(fill=tk.X, pady=10)
        
        self.command_var = tk.StringVar()
        command_entry = ttk.Entry(preview_frame, textvariable=self.command_var, state="readonly")
        command_entry.pack(fill=tk.X, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Help button
        help_button = ttk.Button(button_frame, text="Help", command=self.show_help)
        help_button.pack(side=tk.LEFT, padx=5)
        
        # Update preview button
        update_button = ttk.Button(button_frame, text="Update Preview", command=self.update_preview)
        update_button.pack(side=tk.LEFT, padx=5)
        
        # Run button
        run_button = ttk.Button(button_frame, text="Run Simulation", 
                              command=self.run_simulation, style="Run.TButton")
        run_button.pack(side=tk.RIGHT, padx=5)
        
        # Configure the run button style
        self.style.configure("Run.TButton", background=button_bg, foreground=button_fg, font=('Arial', 10, 'bold'))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initial update of command preview
        self.update_preview()
        
        # Tooltips
        self.create_tooltips()
        
        # Bind to update preview when parameters change
        for var in [self.scenario_var, self.particles_var, self.timestep_var, 
                   self.record_var, self.output_var, self.multiprocessing_var, 
                   self.energy_var, self.fps_var]:
            var.trace("w", lambda *args: self.update_preview())
    
    def create_tooltips(self):
        """Create tooltips for UI elements."""
        # Would implement tooltips here with a custom tooltip class
        # For now, we'll skip this as it requires additional implementation
        pass
    
    def browse_output(self):
        """Open file dialog to select output directory."""
        directory = filedialog.askdirectory(initialdir=os.getcwd())
        if directory:
            self.output_var.set(directory)
            self.update_preview()
    
    def update_preview(self):
        """Update the command preview."""
        script_path = os.path.join(os.path.dirname(__file__), "particle_simulator.py")
        
        cmd = [sys.executable, script_path]
        
        # Add options
        cmd.extend(["--scenario", self.scenario_var.get()])
        cmd.extend(["--particles", str(self.particles_var.get())])
        cmd.extend(["--timestep", str(self.timestep_var.get())])
        
        if self.record_var.get():
            cmd.append("--record")
            cmd.extend(["--output", self.output_var.get()])
        
        if self.multiprocessing_var.get():
            cmd.append("--multiprocessing")
        
        if self.energy_var.get():
            cmd.append("--energy")
            
        # Show the command as a string in the preview
        self.command_var.set(" ".join(cmd))
    
    def run_simulation(self):
        """Run the simulation with current settings."""
        try:
            self.update_preview()
            cmd = self.command_var.get().split()
            
            # Update status
            self.status_var.set("Starting simulation...")
            self.root.update()
            
            # Run the simulation in a new process
            process = subprocess.Popen(cmd)
            
            # Update status
            self.status_var.set("Simulation running. Close simulation window when done.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {str(e)}")
            self.status_var.set("Error starting simulation.")
    
    def show_help(self):
        """Show help information."""
        help_text = """
        Particle Simulator Help
        
        Scenario Types:
        - orbit: A central mass with orbiting particles
        - random: Randomly distributed particles
        - collision: Two clusters of particles set to collide
        - galaxy: A spiral galaxy-like formation
        
        Parameters:
        - Number of Particles: More particles = more complex simulation but slower
        - Time Step: Smaller values give more accurate physics but slower simulation
        - Record Frames: Save each frame as an image in the output directory
        - Use Multiprocessing: Use multiple CPU cores (faster for many particles)
        - Show Energy Conservation: Display energy metrics in real-time
        
        Controls in Simulation:
        - ESC: Exit simulation
        - SPACE: Pause/Resume
        - S: Save screenshot (if recording enabled)
        - C: Toggle controls display
        - R: Reset simulation with same scenario
        - E: Toggle energy display
        - +/-: Increase/decrease time step
        """
        
        # Create a help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Simulation Help")
        help_window.geometry("500x500")
        help_window.minsize(400, 400)
        
        # Add text widget with scrollbar
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=text_widget.yview)
        
        # Close button
        close_button = ttk.Button(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=10)


if __name__ == "__main__":
    # Create and run the application
    root = tk.Tk()
    app = ParticleSimulatorGUI(root)
    root.mainloop()