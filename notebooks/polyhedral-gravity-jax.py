import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    # Prevent memory pre-allocation for flexible memory management
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' # Use 80% of GPU memory default is '.75'

    import marimo as mo
    import numpy as np
    from polyhedral_gravity import Polyhedron, GravityEvaluable, evaluate, PolyhedronIntegrity, NormalOrientation, MetricUnit

    import jax 
    import jax.numpy as jnp

    import sys

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_root)
    from script import mesh_plotting
    from polyhedral_gravity.jax import evaluate as jax_evaluate

    jax.devices()
    return (
        GravityEvaluable,
        MetricUnit,
        NormalOrientation,
        Polyhedron,
        PolyhedronIntegrity,
        jax_evaluate,
        jnp,
        np,
    )


@app.cell
def _(MetricUnit, NormalOrientation, Polyhedron, PolyhedronIntegrity):
    asteroid = Polyhedron(
      polyhedral_source=["bodies/MaterialNodeTreeRocks_1.stl"],
      density=1.0,
      normal_orientation=NormalOrientation.INWARDS, # OUTWARDS (default) or INWARDS
      integrity_check=PolyhedronIntegrity.HEAL,   # VERIFY (default), DISABLE or HEAL
      metric_unit=MetricUnit.METER,    
    )
    return (asteroid,)


@app.cell
def _(GravityEvaluable, asteroid, np):
    computation_points = np.array([[0,0,0],[1,1,1], [2,2,2], [3,3,3]])

    evaluable = GravityEvaluable(polyhedron=asteroid) # stores intermediate computation steps
    return computation_points, evaluable


@app.cell
def _(computation_points, evaluable):
    results = evaluable(
      computation_points=computation_points,
      parallel=True,
    )
    return


@app.cell
def _(jax_evaluate, jnp):
    cube_vertices = jnp.array([
        [-1, -1, -1], [1, -1, -1], [1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [1, -1,  1], [1,  1,  1], [-1,  1,  1],
    ])

    cube_faces = jnp.array([
        [1, 3, 2], [0, 3, 1], [0, 1, 5], [0, 5, 4],
        [0, 7, 3], [0, 4, 7], [1, 2, 6], [1, 6, 5],
        [2, 3, 6], [3, 7, 6], [4, 5, 6], [4, 6, 7],
    ])

    density = 1.0  # kg/m^3

    pts_t = jnp.array([[2.5, 2.5, 2.5]])

    potential_jax, accel_jax, _ = jax_evaluate(cube_vertices, cube_faces, density, pts_t)

    print(f"Potential:    {potential_jax.item():.6e} m^2/s^2")
    print(f"Acceleration: {accel_jax.squeeze().tolist()} m/s^2")
    return (density,)


@app.cell
def _(asteroid, computation_points, jnp):
    asteroid_vertices = jnp.array(asteroid.vertices)
    asteroid_faces = jnp.array(asteroid.faces)
    jax_points = jnp.asarray(computation_points)
    return asteroid_faces, asteroid_vertices, jax_points


@app.cell
def _(asteroid_faces, asteroid_vertices, density, jax_evaluate, jax_points):
    _, accel, _ = jax_evaluate(asteroid_vertices, asteroid_faces, density, jax_points)
    print(accel)
    return


if __name__ == "__main__":
    app.run()
