# Snake-Robot-OGA-ML-Path-Planning
Python-based snake robot simulator integrating forward kinematics, ML-based steering, OGA*, Dijkstra, RRT and RRT* for multi-destination path planning, static/dynamic obstacle avoidance, singularity, workspace and motion analysis.
# Snake Robot OGA*-ML Path Planning Simulator

A Python-based simulation framework for kinematics, machine-learning-assisted steering, multi-destination path planning, full-body obstacle avoidance, singularity analysis, workspace evaluation and dynamic replanning of a planar snake robot.

The simulator was developed as a research platform for evaluating machine learning and classical/evolutionary-style path-planning approaches under progressively challenging navigation conditions.

The major contribution is an Orientation and Geometry-Aware A* planner, referred to as OGA*, which incorporates directional continuity, full-body obstacle inflation, local-turn constraints, a turn penalty and a no-retrace condition for snake-robot navigation.

---

## Research Objective

The project investigates how machine-learning-based steering commands can be integrated with global path-planning algorithms for safe navigation of a hyper-redundant snake robot.

The framework evaluates:

- forward kinematics of a serial snake robot
- machine-learning-based steering prediction
- ordered multi-destination navigation
- obstacle-free path planning
- static obstacle avoidance
- dynamic obstacle avoidance
- online replanning
- full-body collision checking
- workspace analysis
- singularity analysis
- head-position error
- displacement, velocity and acceleration
- planner and ML performance comparison

---

## Robot Configuration

The simulator supports:

- 6 to 16 degrees of freedom
- revolute joints
- planar snake-robot kinematics
- user-defined joint angles
- global and local coordinate frames
- X-axis: red
- Y-axis: green
- Z-axis: blue
- head-led path execution
- full serial-body visualisation

The archived experimental study used a 9-DOF configuration.

### Experimental Destinations

The principal ordered destination points were:

```text
D1 = (400, 300)
D2 = (800, 0)
D3 = (200, -300)
