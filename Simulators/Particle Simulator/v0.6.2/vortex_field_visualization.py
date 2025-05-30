def draw_field_debug(self):
    """Draw debug visualization of the vortex field."""
    if not self.show_field_debug:
        return
        
    # Get all protons
    protons = [p for p in self.particles if p['type'] == 'proton']
    if not protons:
        return
        
    for proton in protons:
        # Calculate expected orbital radius for this proton
        expected_radius = self.calculate_expected_orbital_radius(proton)
        shell_width = expected_radius * 0.15  # Match the STABLE_ORBIT_WIDTH constant
        
        # Draw the vortex field strength at different distances
        samples = 30  # More samples for smoother visualization
        max_distance = expected_radius * 2
        
        for i in range(1, samples + 1):
            radius = i * max_distance / samples
            
            # Calculate relative deviation from expected radius
            deviation = abs(radius - expected_radius) / shell_width
            
            # Calculate field strength - uses same formula as physics_core
            field_strength = np.exp(-0.5 * deviation**2)
            
            # Draw the field with color intensity based on strength
            field_vertices = self.create_circle_vertices(proton['pos'], radius, 40)
            
            # Color gradient based on field strength and distance
            alpha = field_strength * 0.2  # Semi-transparent
            
            if radius < expected_radius - shell_width:
                # Inside inner shell - red gradient
                color = [1.0, 0.0, 0.0, alpha]
            elif radius > expected_radius + shell_width:
                # Outside outer shell - blue gradient
                color = [0.0, 0.0, 1.0, alpha]
            else:
                # Within shell boundaries - green/yellow gradient
                green = min(1.0, field_strength * 1.5)
                color = [field_strength * 0.5, green, 0.0, alpha]
            
            field_colors = np.ones((len(field_vertices), 4), dtype=np.float32) * np.array(color, dtype=np.float32)
            
            self.circle_program['a_position'] = field_vertices
            self.circle_program['a_color'] = field_colors
            self.circle_program.draw('line_loop')