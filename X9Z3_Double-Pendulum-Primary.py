from vpython import *
#Web VPython 3.2
"""
Created by Maximillian DeMarr.
Future work ideas:
- Spring based physics system with constant downward forcing vector
- n-node pendulum incorporating spring chains
- Greater diagnostic control over the graphs
- Fix damping on the coupled term

Last major update 2025/05/21.
"""

# ================================= Canvas =====================================

scene = display(width = 600, height = 600, align='left')
scene.bind('mousedown', move_node)
scene.bind('mouseup', release_mouse_1)

# Create reference coordinate axes
# x-axis (red)
arrow(pos=vec(0,0,0), axis=vec(2,0,0), color=color.white, 
        shaftwidth=0.005, headwidth=0.05)
arrow(pos=vec(0,0,0), axis=vec(-2,0,0), color=color.white,
        shaftwidth=0.005, headwidth=0.05)
label(pos=vec(2,0,0), text='x', color=color.red,
        xoffset=2, yoffset=2, box=False)

# y-axis (green)
arrow(pos=vec(0,0,0), axis=vec(0,2,0), color=color.white,
        shaftwidth=0.005, headwidth=0.05)
arrow(pos=vec(0,0,0), axis=vec(0,-2,0), color=color.white,
        shaftwidth=0.005, headwidth=0.05)
label(pos=vec(0,2,0), text='y', color=color.green,
        xoffset=2, yoffset=2, box=False)

about_me = """
            Click <b>Running</b> to Pause/Play. Adjust simulation accuracy via the 
    <b>Simulation step</b> dropdown. Left-click and drag a mass to reposition it 
    (best while paused). Rotate the view with right-click + drag, zoom with 
    scroll wheel.

    <b>Show graphs</b> impacts performance but offers useful features. For 
    detailed analysis, switch <b>Fast plots</b> to <b>High detail</b> 
    (more resource-intensive). If the pendulum spins too fast, increase 
    <b>Damping</b> or reload the page. For best performance, use Chrome.

    ┌── Physics Discussion ──┐

    The Equations of Motion are derived using the Euler-Lagrange Differential
    Equation. Once we obtain expressions for \\(\\ddot{\\theta}_1\\) and  
    \\(\\ddot{\\theta}_2\\), we apply a forward-Euler integration scheme to 
    compute positions and angular velocities in real time. This method trades 
    accuracy for performance, which suits our purpose here.

    Angles follow the unit circle convention, where \\(\\theta=0\\) points along 
    the positive x-axis (not vertical). If replicating this system, be careful—this 
    orientation alters the expressions for x and y positions, resulting in 
    distinct equations of motion. Subscripts: 1 = blue mass, 2 = red mass.

    <b>Energy Equations:</b>
    \\[T = \\frac{1}{2}m_1(\\ell_1^2 \\dot{\\theta}_1^2) + \\frac{1}{2}m_2\\left( 
    \\ell_1^2 \\dot{\\theta}_1^2 + \\ell_2^2 \\dot{\\theta}_2^2 + 
    2\\ell_1\\ell_2\\dot{\\theta}_1\\dot{\\theta}_2\\cos\\Delta\\theta \\right)\\]
    \\[V = -m_1g\\ell_1\\cos\\theta_1 - m_2g\\left( 
    \\ell_1\\cos\\theta_1 + \\ell_2\\cos\\theta_2 \\right)\\]

    <b>Angular Accelerations:</b>
    \\[\\ddot{\\theta}_1 = \\frac{m_2g\\sin\\theta_2\\cos\\Delta\\theta - 
    m_2\\ell_2\\dot{\\theta}_2^2\\sin\\Delta\\theta - 
    (m_1+m_2)g\\sin\\theta_1}{\\ell_1(m_1 + m_2\\sin^2\\Delta\\theta)}\\]
    \\[\\ddot{\\theta}_2 = \\frac{(m_1+m_2)(\\ell_1\\dot{\\theta}_1^2\\sin\\Delta\\theta - 
    g\\sin\\theta_2 + g\\sin\\theta_1\\cos\\Delta\\theta)}
    {\\ell_2(m_1 + m_2\\sin^2\\Delta\\theta)}\\]
    where \\(\\Delta\\theta = \\theta_1 - \\theta_2\\).

                -<i>Created by Maximillian DeMarr</i>
"""

scene.caption = about_me
MathJax.Hub.Queue(["Typeset", MathJax.Hub, scene.caption])  # LaTeX formatting










# ================================ Methods =====================================

def plot_handler(time, m1, m2, max_data_length=1e4):
    """
    Handles energy data collection and visualization for the double pendulum system.
    
    Tracks and plots:
    - Potential Energy (PE)
    - Kinetic Energy (KE)
    - Total Energy (PE + KE)
    - Phase space trajectory (angle vs angular velocity)
    
    Energy Conservation Check:
    ΔE = |(KE + PE) - E_initial| should be small when damping=0
    
    Args:
        time (float): Current simulation time
        m1 (Node): First pendulum node
        m2 (Node): Second pendulum node
        max_data_length (int): Maximum data points to store before reset
        
    Global Dependencies:
        potential_gcurve: Graph curve for potential energy
        kinetic_gcurve: Graph curve for kinetic energy
        total_energy_gcurve: Graph curve for total energy
        phasor_gcurve: Graph curve for phase space plot
    """
    global energy_graph, potential_gcurve, kinetic_gcurve, total_energy_gcurve
    global phase_plane_graph, phasor_gcurve, fast_plots, gravity
    
    # Clear data if buffer full
    if len(kinetic_gcurve.data) > max_data_length or time > energy_graph.xmax:
        energy_graph.xmax = time + 10  # Extend x-axis
        energy_graph.xmin = time
        try:
            potential_gcurve.data = []
            kinetic_gcurve.data = []
            total_energy_gcurve.data = []
        except:
            # A weird bug has been occurring here with a type error associated with assigning
            # kinetic_gcurve.data = []? I think it is a JavaScript async issue which I cannot
            # easily avoid so I will leave these try/except crutches here to catch it.
            pass 
    elif len(phasor_gcurve.data) > max_data_length:
        phasor_gcurve.data = []

    pe = calculate_potential_energy(m1, m2, g=gravity)
    ke = calculate_kinetic_energy(m1, m2)
    total = pe + ke
    
    # Plot energy data
    potential_gcurve.plot(time, pe)
    kinetic_gcurve.plot(time, ke)
    total_energy_gcurve.plot(time, total)
    
    # Plot phase space (angle vs angular velocity of mass 2)
    phasor_gcurve.plot(m2.angle, m2.angular_velocity)


def calculate_potential_energy(m1, m2, g=9.81):
    """
    Calculates the total gravitational potential energy of a double pendulum system.
    
    The potential energy is calculated relative to the equilibrium position (pendulum hanging straight down),
    summing contributions from both masses:
    
    PE = m₁gh₁ + m₂gh₂
    
    Where:
    h₁ = ℓ₁(1 + sinθ₁)          # Height of mass 1
    h₂ = ℓ₁(1 + sinθ₁) + ℓ₂(1 + sinθ₂)  # Height of mass 2
    
    Visual explanation:
      ▲
      │ ℓ₁(1+sinθ₁) 
      m₁─────┐
             │ ℓ₂(1+sinθ₂)
             m₂
             
    Args:
        m1 (Node): First pendulum node with properties:
            - mass (kg)
            - rod_length (m)
            - angle (rad)
        m2 (Node): Second pendulum node
        gravity (float): Gravitational acceleration (default 9.81 m/s²)
        
    Returns:
        float: Total potential energy in joules
    """
    h1 = m1.rod_length * (1 + sin(m1.angle))
    h2 = h1 + m2.rod_length * (1 + sin(m2.angle))
    return g * (m1.mass * h1 + m2.mass * h2)


def calculate_kinetic_energy(m1, m2):
    """
    Calculates the total kinetic energy of a double pendulum system.
    
    The kinetic energy consists of:
    1. Rotational energy of each mass
    2. Coupling term between the masses
    
    KE = ½m₁v₁² + ½m₂v₂²
    
    Where velocities are:
    v₁² = (ℓ₁ω₁)²
    v₂² = (ℓ₁ω₁)² + (ℓ₂ω₂)² + 2ℓ₁ℓ₂ω₁ω₂cos(θ₁-θ₂)
    
    Expanded form:
    KE = ½m₁(ℓ₁ω₁)² + ½m₂[(ℓ₁ω₁)² + (ℓ₂ω₂)² + 2ℓ₁ℓ₂ω₁ω₂cos(Δθ)]
    
    Args:
        m1 (Node): First pendulum node with properties:
            - mass (kg)
            - rod_length (m)
            - angle (rad)
            - angular_velocity (rad/s)
        m2 (Node): Second pendulum node
            
    Returns:
        float: Total kinetic energy in joules
    """
    # Individual terms
    term1 = m1.mass * (m1.rod_length * m1.angular_velocity)**2
    term2 = m2.mass * (m1.rod_length * m1.angular_velocity)**2
    term3 = m2.mass * (m2.rod_length * m2.angular_velocity)**2
    coupling = 2 * m2.mass * (
        m1.rod_length * m2.rod_length * 
        m1.angular_velocity * m2.angular_velocity *
        cos(m1.angle - m2.angle)
    )
    
    return 0.5 * (term1 + term2 + term3 + coupling)








# ============================= Widget Methods =================================

def move_node(evt):
    """
    Handles interactive node movement during mouse drag events in a hierarchical node graph.
    
    This function enables real-time dragging of sphere nodes during simulation while 
    maintaining the graph's hierarchical structure. When a node is dragged:
    - Its position updates continuously to follow mouse movement
    - All nodes maintain their edge length to prevent compression and extension
    - Angular physics properties are reset to prevent unintended rotation
    - Connecting edges are dynamically redrawn
    
    Movement follows these constraints:
    1. Nodes move in the x-y plane based on projected mouse position
    2. Parent node position is considered rigid
    3. Child nodes follow their parent while preserving their relative direction
    4. Angle calculations recalculated to sync the new vector position to the angle value
    
    Args:
        evt: Mouse event object (required by event handler signature but unused internally)
        
    Global Dependencies:
        mouse_1_up (bool): Tracks left mouse button state (True when released)
        paused (bool): Indicates whether physics simulation is paused
        
    Notes:
        - Operates in a continuous loop at 60Hz while mouse button is pressed
        - Uses vector projection onto the x-y plane (z=0)
        - Maintains angle signage for proper orientation (negative when below parent). This
            is a requirement because the diff_angle() method is strictly zero or positive
        - Explicitly zeros angular velocity/acceleration to prevent drift
    """
    global mouse_1_up, paused
    mouse_1_up = False
    hit = scene.mouse.pick  # Get the object under mouse cursor

    if isinstance(hit, sphere):
        while True:
            rate(60)
            if mouse_1_up:  # Exit condition: mouse button released
                break
            
            parent = hit.node.parent
            # Project mouse position to x-y plane and calculate new position
            hit.pos = parent.pos + norm(scene.mouse.project(normal=vec(0,0,1), point=vec(1,0,0)) - parent.pos)
            
            # Reset rotation physics
            hit.node.angular_velocity = 0
            hit.node.angular_acceleration = 0

            # Calculate and maintain angle with proper signage
            if hit.pos.y < hit.node.parent.pos.y:
                # TODO: Instead of using diff_angle just use the reverse trignometrics
                hit.node.angle = -diff_angle(hit.pos - parent.pos, vec(1,0,0))
            else:
                hit.node.angle =  diff_angle(hit.pos - parent.pos, vec(1,0,0))
                
            hit.node.redraw()  # Update visual connections

            # Propagate movement to child node if exists
            child = hit.node.child
            if child is not None:
                child.pos = hit.pos + norm(child.pos - child.node.parent.pos)

                # Maintain child's angle with proper signage
                if child.pos.y < hit.pos.y:
                    child.node.angle = -diff_angle(child.pos - hit.pos, vec(1,0,0))
                else:
                    child.node.angle = diff_angle(child.pos - hit.pos, vec(1,0,0))
                
                # Handle physics when paused
                if paused:
                    child.node.angular_velocity = 0
                    child.node.angular_acceleration = 0
                
                child.node.redraw()  # Update child's visual connections


def release_mouse_1(evt):
    """
    Handles mouse button release events for node interaction.
    
    Sets a global flag to indicate the left mouse button has been released,
    which terminates any active node dragging operations in the move_node function.
    
    Args:
        evt: The mouse event object (unused but required for event handler signature)
        
    Global Dependencies:
        mouse_1_up (bool): Flag tracking left mouse button state (True when released)
    """
    global mouse_1_up
    mouse_1_up = True


def simulation_pause_run(evt):
    """
    Toggles the physics simulation between paused and running states.
    
    Updates both the simulation state and the triggering button's appearance
    to provide clear visual feedback about the current mode.
    
    Args:
        evt: The button event object triggering the toggle
        
    Global Dependencies:
        paused (bool): Tracks simulation pause state
        
    Effects:
        - Toggles global paused state
        - Updates button text and color (red when paused)
    """
    global paused
    paused = not paused
    if paused:
        evt.text = "Paused"
        evt.textcolor = color.red
    else:
        evt.text = "Running"
        evt.textcolor = color.black


def simulation_speed(evt):
    """
    Adjusts the simulation speed multiplier based on slider input.
    
    Updates both the visual speed indicator and the actual time step
    calculation used by the physics engine.
    
    Args:
        evt: The slider event containing the new speed value
        
    Global Dependencies:
        time_step (float): Base physics time step
        accuracy_menu (menu): Reference to accuracy selection UI
        speed_text (wtext): UI element displaying current speed
        
    Effects:
        - Updates speed display text with formatted value
        - Recalculates effective time step as accuracy * speed value
    """
    global time_step, accuracy_menu
    if evt.obj_type == 'slider':
        evt.value_winput_obj.text = evt.value
        time_step = float(accuracy_menu.selected) * evt.value
    elif evt.obj_type == 'winput':
        time_step = float(accuracy_menu.selected) * float(evt.text)
        


def simulation_accuracy(evt):
    """
    Adjusts simulation accuracy parameters based on dropdown selection.
    
    This function updates multiple physics and visualization parameters:
    - Base time step for physics calculations
    - Simulation update rate
    - Graphing refresh interval
    - Node visualization update interval
    
    Args:
        evt: The dropdown event containing the new accuracy value
        
    Global Dependencies:
        time_step (float): Base physics time step
        update_rate (int): Physics engine refresh rate
        graphing_period (int): Data sampling interval for graphs
        null_node (Node): Root node of the simulation
        
    Effects:
        - Updates all time-dependent simulation parameters
        - Propagates new interval settings to the terminal node through node hierarchy
    """
    global time_step, update_rate, graphing_period, null_node
    time_step = float(evt.selected)
    update_rate = abs(int(1/time_step))
    graphing_period = int(update_rate/40)  # Fine-tuned
    
    node = null_node
    while node.child is not None:
        node = node.child.node
    node.obj.interval = int(update_rate/200)  # Fine-tuned


def simulation_dampening(evt):
    """
    Adjusts the energy damping coefficient based on slider input.
    
    Updates both the visual damping indicator and the physical damping
    coefficient used in the simulation. The damping value is applied
    as a negative coefficient to represent energy loss. This value strictly
    scales with the velocity of the nodes.
    
    Args:
        evt: The slider event containing the new damping value
        
    Global Dependencies:
        damping_coefficient (float): Energy loss factor
        dampener_text (wtext): UI element displaying current damping
        
    Effects:
        - Updates damping display text with formatted value
        - Sets global damping coefficient (stored as negative value)
    """
    global damping_coefficient
    if evt.obj_type == 'slider':
        evt.value_winput_obj.text = -evt.value
        damping_coefficient = -evt.value
    elif evt.obj_type == 'winput':
        value = float(evt.text)
        if value <= 0:
            damping_coefficient = value
        else:
            evt.text = -value
            damping_coefficient = -value


def about_me_toggle(evt):
    """
    Toggles the display of an informational 'About Me' section with dynamic layout adjustments.
    
    This function manages:
    - Toggling between full scene view and about me view
    - Adjusting scene width based on current display mode
    - Updating button visual feedback
    - Rendering LaTeX-formatted content when displayed
    
    Args:
        evt: Event object triggering the toggle
        
    Global Dependencies:
        about_me (str): Content to display in the about me section
        caption_enabled (bool): Tracks if about me content is currently displayed
        about_me_button (button): Reference to the UI button triggering this function
        graphs_enabled (bool): Tracks if graph visualization is active
        
    Effects:
        - Modifies scene width and caption content
        - Updates button background color as visual feedback
        - Triggers MathJax rendering for LaTeX content
    """
    global about_me, caption_enabled, about_me_button, graphs_enabled
    if caption_enabled:
        caption_enabled = not caption_enabled
        about_me_button.text = "Explain me!"
        scene.width = 600 if graphs_enabled else 1300
        scene.caption = ''
    else:
        if graphs_enabled:
            canvas_graphs_toggle(evt=None)
        caption_enabled = not caption_enabled
        about_me_button.text = "Hide words "
        scene.width = 600
        scene.caption = about_me
        MathJax.Hub.Queue(["Typeset", MathJax.Hub, scene.caption])


def canvas_graphs_toggle(evt):
    """
    Toggles the display of energy and phase plane graphs with associated UI updates.
    
    Manages the complete lifecycle of physics visualization graphs including:
    - Creation and destruction of energy and phase plane plots
    - UI state management (button colors, scene dimensions)
    - Coordination with other display elements (about me section)
    - Initialization of data collection curves
    
    When enabled:
    - Creates two side-by-side graphs (energy vs time and phase plane)
    - Initializes three energy curves (potential, kinetic, total)
    - Initializes phase plane trajectory curve
    - Adjusts scene layout to accommodate graphs
    
    When disabled:
    - Removes all graph visualizations
    - Resets UI elements to default state
    
    Args:
        evt: The button event object triggering the toggle (required but unused)
        
    Global Dependencies:
        graphs_enabled (bool): Tracks current graph display state
        caption_enabled (bool): Tracks about me section visibility
        graphing_button (button): Reference to the triggering UI button
        Various graph and curve objects (potential_gcurve, kinetic_gcurve, etc.)
        
    Effects:
        - Toggles graphs_enabled flag
        - Updates button visual feedback (gray=active, white=inactive)
        - Adjusts scene width to accommodate/remove graphs
        - Manages graph object lifecycle (creation/deletion)
        - Coordinates with about_me_toggle to handle conflicting displays
        - Initializes/resets all plot data curves
    """
    global graphs_enabled, caption_enabled, graphing_button, potential_gcurve, kinetic_gcurve
    global total_energy_gcurve, phasor_gcurve, energy_graph, phase_plane_graph, fast_plots
    graphs_enabled = not graphs_enabled

    if graphs_enabled:
        if caption_enabled:
            about_me_toggle(evt)  # Ensure about me doesn't overlap with graphs
            
        graphing_button.text = "Hide graphs"  # Visual active state
        scene.width = 600  # Adjust scene width for graph layout

        # Initialize energy plot with three tracked quantities
        energy_graph = graph(
            align='right', width=750, height=280,
            title='<b>Total Energy vs Time:</b>',
            xtitle="Time (s)", ytitle="Energy (J)",
            scroll=False, fast=fast_plots, xmax=10,
            background=color.gray(0.95),

        )
        potential_gcurve =  gcurve(color=color.blue, label="Potential", width=1)
        kinetic_gcurve = gcurve(color=color.red, label="Kinetic", width=1)
        total_energy_gcurve = gcurve(color=color.green, label="Total", width=1)
        
        # Initialize phase plane plot
        phase_plane_graph = graph(
            align='right', width=750, height=280,
            title='<b>Phase plane:</b>',
            xtitle="Theta 2 (rad)", ytitle="Angular Velocity 2 (rad/s)",
            scroll=False, fast=fast_plots,
            background=color.black
        )
        phasor_gcurve = gcurve(color=vec(1,0.7,0.4), width=1)

    else:
        graphing_button.text = "Show graphs"  # Visual inactive state
        energy_graph.delete()  # Clean up energy plot
        phase_plane_graph.delete()  # Clean up phase plane plot
        potential_gcurve.data = []  # Wipe old data  ---
        kinetic_gcurve.data = []
        total_energy_gcurve.data = []
        phasor_gcurve.data = []


def fast_plotting_toggle(evt):
    global fast_plots
    fast_plots = not fast_plots
    if fast_plots:
        evt.text = "Fast plots "
        evt.textcolor = color.black
    else:
        evt.text = "High detail"
        evt.textcolor = color.red


def change_gravity(evt):
    global gravity
    g = 9.81  # m/s²
    if evt.obj_type == 'slider':
        gravity = g * evt.value
        evt.value_winput_obj.text = evt.value
    elif evt.obj_type == 'winput':
        gravity = g * evt.text


def change_rod_length(evt):
    if evt.obj_type == 'slider':
        evt.respective_node.rod_length = evt.value
        evt.value_winput_obj.text = evt.value
    elif evt.obj_type == 'winput':
        evt.respective_node.rod_length = float(evt.text)
    evt.respective_node.redraw()


def change_mass(evt):
    if evt.obj_type == 'slider':
        evt.respective_node.mass = evt.value
        evt.value_winput_obj.text = evt.value
    elif evt.obj_type == 'winput':
        evt.respective_node.mass = float(evt.text)
    evt.respective_node.redraw()











# ================================ Classes =====================================

class Node:
    """
    A physics-based pendulum node with hierarchical connections and Euler-integrated dynamics.
    
    Represents a mass point in a chain pendulum system with:
    - Physical properties (mass, length, angular state)
    - Visual 3D representation (sphere + connecting cylinder)
    - Parent-child relationships forming pendulum chain

    Example Implementation:
        null_node = Node(position=vec(0,0,0), kwargs={'radius':default_radius, 'visible':False})
        mass_1 = Node(position=vec(1,0,0), kwargs={'radius':default_radius, 'color':color.cyan})
        mass_2 = Node(position=vec(2,0,0), kwargs={
            'radius':default_radius, 
            'color':vec(1,0.7,0.2), 
            'make_trail':True, 
            'trail_radius':default_radius/20, 
            'retain':1000, 
            'interval':500
        })
        null_node.link(mass_1, kwargs={'color':color.blue})
        mass_1.link(mass_2, kwargs={'color':color.red})

    Attributes:
        obj (sphere): Primary visual object (position tied to physics state)
        edge_obj (cylinder): Visual connection to child node (None if terminal)
        parent (sphere): Parent node's visual object (None if root)
        child (sphere): Child node's visual object (None if terminal)
        angle (float): Angular displacement from vertical [rad]
        angular_velocity (float): Current angular velocity [rad/s]
        angular_acceleration (float): Current angular acceleration [rad/s²]
        mass (float): Node mass [kg] (default: 1)
        rod_length (float): Distance to parent node [m] (auto-set during linking)
    """
    
    def __init__(self, position=vec(0,0,0), kwargs={}):
        """
        Initialize a pendulum node with optional visual customization.

        Args:
            position (vec): Initial 3D position in world coordinates
            kwargs (dict): Visual properties to apply to the sphere object. Supported options:
                - radius (float): Sphere size [VPython units]
                - color (color/vec): RGB color value
                - visible (bool): Whether to show the object
                - make_trail (bool): Enable position history trail
                - trail_radius (float): Thickness of trail
                - retain (int): Number of trail segments to keep
                - interval (int): Trail update frequency
                - texture: Surface texture object
                - opacity (float): Transparency (0=invisible, 1=opaque)
                - shininess (float): Surface reflectivity
        """
        self.obj = sphere(
            pos=position,
            make_trail=False,
            node=self  # Back-reference for physics updates
        )

        # Apply any visual customization
        for key, val in kwargs.items():
            self.obj[key] = val  # Assume all attributes are valid
        
        # Physics state initialization
        self.angle = diff_angle(vec(1,0,0), position)  # Relative to +x axis
        self.angular_velocity = 0.0
        self.angular_acceleration = 0.0
        self.mass = 10.0  # [kg]
        self.rod_length = 0.0  # Will be set during linking

        # Connection properties
        self.edge_obj = None  # Visual edge to child (cylinder)
        self.parent = None  # Reference to parent sphere
        self.child = None  # Reference to child sphere

    def link(self, other, kwargs={}):
        """
        Establish a parent-child relationship with another node, creating visual connection.
        
        Args:
            other (Node): The node to link as a child of this node
            kwargs (dict): Visual properties for the connecting cylinder. Supports all basic
                cylinder object parameters.
        """
        self.edge_obj = cylinder(
            pos=self.obj.pos,
            axis=other.obj.pos - self.obj.pos,
            radius=self.obj.radius/10  # Default to 1/10th of node size
        )

        # Apply any visual customization
        for key, val in kwargs.items():
            self.edge_obj[key] = val  # Assume all attributes are valid

        # Establish bidirectional references
        self.child = other.obj  # Store pointer to child
        other.parent = self.obj  # Set our pointer as parent of child
        other.rod_length = mag(self.obj.pos - other.obj.pos)  # Auto-set length

    def euler_forward_step(self, dt):
        """
        Advance node physics by one Euler timestep (recursively updates children).
        
        Integration scheme:
        1. ω(t+Δt) = ω(t) + α(t)·Δt
        2. θ(t+Δt) = θ(t) + ω(t)·Δt
        
        Args:
            dt (float): Timestep size [seconds]
        """
        # Update angular kinematics
        self.angular_velocity += self.angular_acceleration * dt
        self.angle += self.angular_velocity * dt
        
        # Propagate to child node if exists
        if self.child:
            self.child.node.euler_forward_step(dt)
        
        self.redraw()  # Update visual representation

    def redraw(self):
        """
        Synchronize visual elements with current physics state.
        
        Updates:
        - Node position based on current angle
        - Connection visuals to parent and child
        """
        if self.parent:
            # Update node position relative to parent
            self.obj.pos = self.parent.pos + self.rod_length * vec(cos(self.angle), sin(self.angle), 0)
            
            # Update parent's connection to us
            self.parent.node.edge_obj.pos = self.parent.pos
            self.parent.node.edge_obj.axis = self.obj.pos - self.parent.pos

        # Update our connection to child if exists
        if self.child:
            self.edge_obj.pos = self.obj.pos
            self.edge_obj.axis = self.child.pos - self.obj.pos

    def load_sliders(self, name_field='Rod'):
        """
        Creates and initializes interactive sliders and associated UI elements for controlling pendulum parameters.
        
        This function sets up two slider controls (rod length and mass) with their corresponding:
        - Title labels
        - Numeric input fields
        - Unit labels
        The sliders are bound to their respective callback functions for real-time updates. The widget population
        order is staggered in the way it is as there is no way to finely control the position/location after the
        object has been made. Hopefully a change soon to come in Web VPython future!
        
        Args:
            name_field (str): Optional prefix text for the pendulum's name in the length slider title.
                            Defaults to 'Rod' if not specified.
        """
        
        # Rod Length Slider Group
        self.rod_length_slider = slider(
            min=0.1, max=10, value=self.rod_length, step=0.1,
            length=180, pos=scene.title_anchor, bind=change_rod_length,
            respective_node=self,
            title_wtext_obj=wtext(
                text=f'\n<b>{name_field} -- Length:</b>',  # Dynamic title with pendulum name
                pos=scene.title_anchor
            ),
            value_winput_obj=None, suffix_wtext_obj=None,  # Will be populated below
            obj_type='slider'
        )
        # Numeric input field for precise rod length control
        self.rod_length_slider.value_winput_obj = winput(
            type='numeric', width=30, height=20,
            respective_node=self,
            text=self.rod_length_slider.value,  # Initialized with current rod length
            pos=scene.title_anchor,
            bind=change_rod_length,  # Same callback as slider for consistent behavior
            obj_type='winput'
        )
        # Unit label for rod length (meters)
        self.rod_length_slider.suffix_wtext_obj = wtext(
            text='m',
            pos=scene.title_anchor
        )

        # Mass Slider Group
        self.mass_slider = slider(
            min=0.01, max=10, value=self.mass,
            length=180, pos=scene.title_anchor, bind=change_mass,
            respective_node=self,
            title_wtext_obj=wtext(
                text='<b>   Mass:</b>',  # Consistent indentation with length slider
                pos=scene.title_anchor
            ),
            value_winput_obj=None, suffix_wtext_obj=None,  # Will be populated below
            obj_type='slider'
        )
        # Numeric input field for precise mass control
        self.mass_slider.value_winput_obj = winput(
            type='numeric', width=30, height=20,
            respective_node=self,
            text=self.mass_slider.value,  # Initialized with current mass
            pos=scene.title_anchor,
            bind=change_mass,  # Same callback as slider for consistent behavior
            obj_type='winput'
        )
        # Unit label for mass (kilograms)
        self.mass_slider.suffix_wtext_obj = wtext(
            text='kg',
            pos=scene.title_anchor
        )









# ================================ Simulation Globals ==========================
# Physical parameters
default_radius = 0.1      # Default visual size for all nodes [VPython units]
damping_coefficient = 0   # Energy loss factor [dimensionless]
gravity = 9.81            # Gravitational acceleration [m/s²]

# Time management
time_step = 1e-5          # Base physics timestep [seconds]
time = 0                  # Cumulative simulation time [seconds]
update_rate = 1/time_step # Physics engine refresh rate [Hz]

# State flags
mouse_1_up = True         # Tracks left mouse button state (True=released)
paused = False            # Simulation pause state
engine_running = True     # Master control for simulation loop
caption_enabled = True    # Toggles explanatory text display
graphs_enabled = False    # Toggles energy/phase graphs
fast_plots = True         # Toggles fast plotting feature
advanced_editing = False  # Toggles pendulum editing features

# Root node of pendulum system (invisible anchor point)
null_node = Node(position=vec(0,0,0), kwargs={'radius':default_radius, 'visible':False})








# ============================== Interactive Widgets ==========================
#                                                                             
#   Render Order Map:               Functional Hierarchy:                      
#                                                                             
#   1. ┌────────────────────┐        ┌──────────────────────────────┐         
#      │  SPEED CONTROL     │        │           SYSTEM             │         
#      └────────────────────┘        ├──────────────────────────────┤         
#   2. ┌────────────────────┐        │  ┌────────────────────┐      │         
#      │  DAMPING (b)       │        │  │ • Speed Control    │      │         
#      └────────────────────┘        │  │ • Numerical Accuracy│     │         
#   3. ┌────────────────────┐        │  └────────────────────┘      │         
#      │  NUMERICAL ACCURACY│        ├──────────────────────────────┤         
#      └────────────────────┘        │           PHYSICS            │         
#   4. ┌────────────────────┐        │  ┌────────────────────┐      │         
#      │  ▣ RUN/PAUSE       │        │  │ • Damping (b)      │     │         
#      ├────────────────────┤        │  │ • Gravity (G)      │      │         
#      │  🔍 HIDE TEXT      │        │  └────────────────────┘     │         
#      ├────────────────────┤        ├──────────────────────────────┤         
#      │  📈 SHOW GRAPHS    │        │           UI                │         
#      ├────────────────────┤        │  ┌────────────────────┐      │         
#      │  ⚡ FAST PLOTS     │        │  │ • Run/Pause        │     │         
#      └────────────────────┘        │  │ • Toggles          │      │         
#   5. ┌────────────────────┐        │  └────────────────────┘      │         
#      │  GRAVITY (G)       │        └──────────────────────────────┘         
#      └────────────────────┘                                               
#                                                                           
# ===========================================================================

# [1] SYSTEM: Speed Control (Renders FIRST) ---------------------------------
speed_slider = slider(  # [-1x, 1x] Time scaling (1.0 = real-time)
    min=-1, max=1, value=1.00, length=300, step=0.1,
    pos=scene.title_anchor, bind=simulation_speed,
    title_wtext_obj=wtext(
        text='<b>Simulation speed:</b>',
        pos=scene.title_anchor
    ),
    value_winput_obj=None, suffix_wtext_obj=None,  # Will be populated below
    obj_type='slider'
)
speed_slider.value_winput_obj = winput(
    type='numeric', width=30, height=20,
    text=speed_slider.value,
    pos=scene.title_anchor,
    bind=simulation_speed,
    obj_type='winput'
)
speed_slider.suffix_wtext_obj = wtext(
    text='x',
    pos=scene.title_anchor
)

# [2] PHYSICS: Damping (Renders SECOND) ------------------------------------
dampener_slider = slider(  # [0-1] Damping coefficient 'b' (dimensionless)
    min=0, max=1, step=0.01, value=damping_coefficient,
    length=180, pos=scene.title_anchor, bind=simulation_dampening,
    title_wtext_obj=wtext(
        text='<b>   Damping coefficient (b):</b>',
        pos=scene.title_anchor
    ),
    value_winput_obj=None, suffix_wtext_obj=None,  # Will be populated below
    obj_type='slider'
)
dampener_slider.value_winput_obj = winput(
    type='numeric', width=30, height=20,
    text=dampener_slider.value,
    pos=scene.title_anchor,
    bind=simulation_dampening,
    obj_type='winput'
)

# [3] SYSTEM: Numerical Accuracy (Renders THIRD) ---------------------------
accuracy_menu = menu(  # Default 1e-5 (Log10 steps from 1e-2 to 1e-7)
    choices=['1e-2','1e-3','1e-4','1e-5','1e-6','1e-7'],
    index=3,  # Defaults to 1e-5
    pos=scene.title_anchor,
    bind=simulation_accuracy,
    menu_title=wtext(
        text='<b>\nSimulation step:</b> ',
        pos=scene.title_anchor
    )
)

# [4] UI BUTTONS (Renders FOURTH) ------------------------------------------
run_pause_button = button(   # ▣/▶ Toggles simulation execution
    text="Running",
    pos=scene.title_anchor,
    bind=simulation_pause_run,
    button_title=wtext(
        text='  ',
        pos=scene.title_anchor
    )
)
about_me_button = button(    # 🔍 Toggles descriptive text
    text='Hide words ',
    pos=scene.title_anchor,
    bind=about_me_toggle,
    button_title=wtext(
        text='  ',
        pos=scene.title_anchor
    )
)
graphing_button = button(    # 📈 Toggles graph visibility
    text="Show graphs",
    pos=scene.title_anchor,
    bind=canvas_graphs_toggle,
    button_title=wtext(
        text='  ',
        pos=scene.title_anchor
    )
)
fast_plots_button = button(  # ⚡/⏳ Toggles performance vs quality
    text="Fast plots",
    pos=scene.title_anchor,
    bind=fast_plotting_toggle,
    button_title=wtext(
        text='  ',
        pos=scene.title_anchor
    )
)

# [5] PHYSICS: Gravity (Renders FIFTH) --------------------------------------
gravity_slider = slider(  # [-10G, 10G] Normalized to Earth gravity
    min=-10, max=10, value=gravity/9.81, step=0.1,
    length=209, pos=scene.title_anchor, bind=change_gravity,
    title_wtext_obj=wtext(
        text='  <b>Gravity:</b>',
        pos=scene.title_anchor
    ),
    value_winput_obj=None, suffix_wtext_obj=None,  # Will be populated below
    obj_type='slider'
)
gravity_slider.value_winput_obj = winput(
    type='numeric', width=30, height=20,
    text=gravity_slider.value,
    pos=scene.title_anchor,
    bind=change_gravity,
    obj_type='winput'
)
gravity_slider.suffix_wtext_obj = wtext(
    text='G',
    pos=scene.title_anchor
)








# ================================== Main Simulation Loop ==============================

def main():
    """
    Core simulation loop for double pendulum physics visualization.
    
    Handles:
    - System initialization (nodes and connections)
    - Real-time physics integration
    - Energy tracking and visualization
    - User interaction via global controls
    
    Physics Overview:
    Implements coupled differential equations for double pendulum motion:
    For mass 1 (inner pendulum):
    
    α₁ = [ -m₁gcosθ₁ 
        + (m₂gcos(θ₁-2θ₂))/2 
        - (m₂gcosθ₁)/2 
        - (ℓ₁m₂sin(2θ₁-2θ₂)ω₁²)/2 
        - (m₂ℓ₂sin(θ₁-θ₂)ω₂² ] 
        / [ ℓ₁(m₁ - m₂cos²(θ₁-θ₂) + m₂ℓ₁ ]
        + bω₁ℓ₁
    
    For mass 2 (outer pendulum):
    α₂ = [ (m₁gcos(2θ₁-θ₂))/2 
       - (m₁gcosθ₂)/2 
        + (m₂gcos(2θ₁-θ₂))/2 
        - (m₂gcosθ₂)/2 
        + m₁ℓ₁sin(θ₁-θ₂)ω₁² 
        + ℓ₁m₂sin(θ₁-θ₂)ω₁² 
        + (m₂ℓ₂sin(2θ₁-2θ₂)ω₂²)/2 ] 
        / [ ℓ₂(m₁ - m₂cos²(θ₁-θ₂) + m₂ℓ₂ ]
        + bω₂ℓ₂
    
    Where:
    θ = angle, ω = angular velocity, α = angular acceleration
    b = damping coefficient, g = gravitational acceleration

    CAUTION: If you are looking to implement these equations yourself, be warned.
    My system defines x = cos(theta) instead of the conventional x = sin(theta).
    The same for y = sin(theta) instead. This is because of of Glowscript handles
    trignometric calculations and this cannot be easily avoided.
    """
    global time, time_step, update_rate, engine_running, paused, gravity
    global damping_coefficient, graphs_enabled, graphing_period, null_node

    # ==================== System Initialization ====================
    # Create pendulum nodes with visual properties
    # For some reason we cannot use more than 1 line when instantiating a custom class
    kwargs_1={
        'radius': default_radius,
        'color': color.cyan
    }
    mass_1 = Node(position=vec(1,0,0), kwargs=kwargs_1)

    kwargs_2={
        'radius': default_radius,
        'color': vec(1,0.7,0.2),  # Orange
        'make_trail': True,
        'trail_radius': default_radius/20,
        'retain': 1000,  # Trail length
        'interval': 500   # Trail update frequency
    }
    mass_2 = Node(position=vec(2,0,0), kwargs=kwargs_2)

    # Establish parent-child relationships
    null_node.link(mass_1, kwargs={'color': color.blue})
    mass_1.link(mass_2, kwargs={'color': color.red})
    
    mass_1.load_sliders(name_field='Rod 1')
    sleep(1)  # Wait for widgets to load in correct position as they are async
    mass_2.load_sliders(name_field='Rod 2')

    # ==================== Simulation Parameters ====================
    iteration = 0
    graphing_period = int(update_rate/40)  # Graph update frequency

    # ==================== Main Physics Loop ====================
    while engine_running:
        rate(update_rate)  # Maintain consistent framerate
        
        if paused:
            continue  # Skip physics when paused

        # Calculate angular accelerations (coupled pendulum equations)
        mass_1.angular_acceleration = (
            (-gravity * mass_1.mass * cos(mass_1.angle) 
            + gravity * mass_2.mass * cos(mass_1.angle - 2*mass_2.angle)/2
            - gravity * mass_2.mass * cos(mass_1.angle)/2
            - mass_1.rod_length * mass_2.mass * sin(2*mass_1.angle - 2*mass_2.angle) * mass_1.angular_velocity**2/2
            - mass_2.mass * mass_2.rod_length * sin(mass_1.angle - mass_2.angle) * mass_2.angular_velocity**2)
            / (mass_1.rod_length * (mass_1.mass 
            - mass_2.mass * cos(mass_1.angle - mass_2.angle)**2 
            + mass_2.mass))
            + damping_coefficient * mass_1.angular_velocity * mass_1.rod_length
        )

        mass_2.angular_acceleration = (
            (gravity * mass_1.mass * cos(2*mass_1.angle - mass_2.angle)/2
            - gravity * mass_1.mass * cos(mass_2.angle)/2
            + gravity * mass_2.mass * cos(2*mass_1.angle - mass_2.angle)/2
            - gravity * mass_2.mass * cos(mass_2.angle)/2
            + mass_1.mass * mass_1.rod_length * sin(mass_1.angle - mass_2.angle) * mass_1.angular_velocity**2
            + mass_1.rod_length * mass_2.mass * sin(mass_1.angle - mass_2.angle) * mass_1.angular_velocity**2
            + mass_2.mass * mass_2.rod_length * sin(2*mass_1.angle - 2*mass_2.angle) * mass_2.angular_velocity**2/2)
            / (mass_2.rod_length * (mass_1.mass 
            - mass_2.mass * cos(mass_1.angle - mass_2.angle)**2 
            + mass_2.mass))
            + damping_coefficient * mass_2.angular_velocity * mass_2.rod_length
        )

        # Update physics state
        mass_1.euler_forward_step(time_step)
        time += time_step
        iteration += 1

        # Update graphs if enabled and period reached
        if graphs_enabled and not (iteration % graphing_period):
            plot_handler(time, mass_1, mass_2)


# Start simulation
main()


# End of Program