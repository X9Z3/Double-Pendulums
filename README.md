# 🌌 Double Pendulum Simulation (Glowscript)

**Created by Maximillian DeMarr**  
_Last major update: 2025-05-21_

Interactive 3D simulation of a double pendulum built using [GlowScript](https://www.glowscript.org/). This project simulates real-time motion based on Lagrangian mechanics and Euler integration.

## 🔗 Live Demo

👉 [Click here to launch the simulation in your browser](https://www.glowscript.org/#/user/X9Z3/folder/X9Z3Publications/program/Double-Pendulum-Primary)  

---

## 🎮 Instructions

- **Click "Running"** to pause or play the simulation.
- Use the **Simulation Step** dropdown to adjust precision (larger steps run faster, less accurately).
- **Drag masses** to reposition them manually (pause first for best results).
- Right-click + drag to **rotate the view**, and use your scroll wheel to zoom.
- Toggle **Show Graphs** for energy tracking (note: high detail may reduce performance).
- If motion becomes erratic, increase **Damping** or reload the page.

---

## 📊 Physics Overview

The system is modeled using the **Euler-Lagrange equations**, with angular accelerations derived symbolically and solved numerically using the forward Euler method.

> ⚠️ This project defines positions in the following way. This is opposite to the conventional vertical orientation, due to how Glowscript handles axes.
```math
x=\cos(\theta) \quad y=\sin(\theta)
```

### 🧮 Energy Equations

**Kinetic Energy:**

```math
T = \frac{1}{2}m_1(\ell_1^2 \dot{\theta}_1^2) + \frac{1}{2}m_2\left(\ell_1^2 \dot{\theta}_1^2 + \ell_2^2 \dot{\theta}_2^2 + 2\ell_1\ell_2\dot{\theta}_1\dot{\theta}_2\cos\Delta\theta \right)
```

**Potential Energy:**

```math
V = -m_1g\ell_1\cos\theta_1 - m_2g\left(\ell_1\cos\theta_1 + \ell_2\cos\theta_2 \right)
```

### 🔁 Angular Accelerations
```math
\ddot{\theta}_1 = \frac{m_2g\sin\theta_2\cos\Delta\theta - 
    m_2\ell_2\dot{\theta}_2^2\sin\Delta\theta - 
    (m_1+m_2)g\sin\theta_1}{\ell_1(m_1 + m_2\sin^2\Delta\theta)}
```
```math
\ddot{\theta}_2 = \frac{(m_1+m_2)(\ell_1\dot{\theta}_1^2\sin\Delta\theta - 
    g\sin\theta_2 + g\sin\theta_1\cos\Delta\theta)}
    {\ell_2(m_1 + m_2\sin^2\Delta\theta)}
```
where
```math
\Delta\theta = \theta_1 - \theta_2.
```


## 🧮 Related Tools

This repo uses:

- [SymPy](https://www.sympy.org/) for symbolic derivation of motion equations → [EOM Parser Repo](https://github.com/X9Z3/Lagrangian-Equation-Solver)
- [GlowScript](https://www.glowscript.org/) for 3D browser-based physics rendering

---

## 🧑‍💻 Author

**Maximillian DeMarr**  
Built with curiosity, calculus, and caffeine ☕.  
For inquiries or collaborations, feel free to reach out!
