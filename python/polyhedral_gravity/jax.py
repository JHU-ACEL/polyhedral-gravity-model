"""
Differentiable polyhedral gravity model - JAX interface.

Implements the Tsoulis (2012) line-integral formula in pure JAX, making
the gravitational potential and acceleration differentiable with respect to
vertex positions and density. The implementation is fully JIT compatible.
"""

import jax
import jax.numpy as jnp

G_SI: float = 6.67430e-11  # m^3 kg^-1 s^-2


@jax.jit
def evaluate(
    vertices: jax.Array,
    faces: jax.Array,
    density: float,
    computation_points: jax.Array,
    gravitational_constant: float = G_SI,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Gravitational potential, acceleration, and gradient tensor for a polyhedron,
    differentiable w.r.t. vertex positions and density.

    Args:
        vertices:                (N, 3) vertex positions [m].
        faces:                   (F, 3) triangle vertex indices, integer dtype.
        density:                 Constant density [kg/m^3].
        computation_points:      (Q, 3) evaluation positions [m].
        gravitational_constant:  Gravitational constant, defaults to 6.67430e-11.

    Returns:
        potential:    (Q,)   gravitational potential [m^2/s^2].
        acceleration: (Q, 3) gravitational acceleration [m/s^2].
        tensor:       (Q, 6) gravity gradient tensor [1/s^2],
                             components ordered as [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz].
    """

    eps = 1e-30

    # face vertex coords: (F, 3, 3)
    face_vertices = vertices[faces]  # [f, vertex_idx, xyz]

    # shift so each query point is at the origin: (Q, F, 3, 3)
    face_vertices_rel = face_vertices[None] - computation_points[:, None, None, :]

    v0 = face_vertices_rel[:, :, 0]  # (Q, F, 3)
    v1 = face_vertices_rel[:, :, 1]
    v2 = face_vertices_rel[:, :, 2]

    # ------------------------------------------------------------------ #
    #  Edge vectors G_pq and outward unit normal (face_normal)
    # ------------------------------------------------------------------ #
    edge0 = v1 - v0  # (Q, F, 3)
    edge1 = v2 - v1
    edge2 = v0 - v2

    edges = jnp.stack([edge0, edge1, edge2], axis=2)  # (Q, F, 3, 3)  [q,f,seg,xyz]
    segment_start_points = jnp.stack(
        [v0, v1, v2], axis=2
    )  # (Q, F, 3, 3)  segment start vertices

    cross_01 = jnp.cross(edge0, edge1)  # (Q, F, 3)
    face_normal = cross_01 / jnp.clip(jnp.linalg.norm(cross_01, axis=-1, keepdims=True), min=eps)

    # ------------------------------------------------------------------ #
    #  Segment unit normals n_pq = (G_pq x face_normal) / |...|
    # ------------------------------------------------------------------ #
    face_normal_expanded = jnp.broadcast_to(face_normal[:, :, None], edges.shape)  # (Q, F, 3, 3)
    n_pq = jnp.cross(edges, face_normal_expanded)
    n_pq = n_pq / jnp.clip(jnp.linalg.norm(n_pq, axis=-1, keepdims=True), min=eps)

    # ------------------------------------------------------------------ #
    #  Signed plane distance, plane normal orientation, projection P'
    # ------------------------------------------------------------------ #
    hp_signed = jnp.sum(face_normal * v0, axis=-1)  # (Q, F)   = N_p . v0
    h_p = jnp.abs(hp_signed)
    sigma_p = jnp.sign(hp_signed)  # sigma_p

    plane_projection = hp_signed[..., None] * face_normal  # (Q, F, 3)  P' = hp_signed * face_normal

    # ------------------------------------------------------------------ #
    #  Segment normal orientations sigma_pq and edge distances h_pq
    # ------------------------------------------------------------------ #
    plane_projection_expanded = jnp.broadcast_to(
        plane_projection[:, :, None], segment_start_points.shape
    )  # (Q, F, 3, 3)
    h_pq_raw = -jnp.sum(
        n_pq * (plane_projection_expanded - segment_start_points), axis=-1
    )  # (Q, F, 3)

    edge_length = jnp.linalg.norm(edges, axis=-1)  # (Q, F, 3)
    edge_scale = jnp.max(edge_length, axis=-1, keepdims=True)  # (Q, F, 1)
    
    h_pq_signed = jnp.where(
        jnp.abs(h_pq_raw) < 1e-10 * edge_scale,
        jnp.zeros_like(h_pq_raw),
        h_pq_raw,
    )
    sigma_pq = jnp.sign(h_pq_signed)  # (Q, F, 3)
    h_pq = jnp.abs(h_pq_signed)  # (Q, F, 3)

    # ------------------------------------------------------------------ #
    #  Projection parameter t along each edge (needed for s1, s2)
    # ------------------------------------------------------------------ #
    edge_length_sq = jnp.clip(jnp.sum(edges * edges, axis=-1), min=eps)  # (Q, F, 3)
    t_pq = jnp.sum((plane_projection_expanded - segment_start_points) * edges, axis=-1) / edge_length_sq

    # ------------------------------------------------------------------ #
    #  Distances from P (origin) to segment endpoints
    # ------------------------------------------------------------------ #
    segment_end_points = jnp.roll(segment_start_points, -1, axis=2)  # v_end = v_{(q+1)%3}
    l1 = jnp.linalg.norm(segment_start_points, axis=-1)  # (Q, F, 3)  |v_start|
    l2 = jnp.linalg.norm(segment_end_points, axis=-1)  # (Q, F, 3)  |v_end|

    # signed 1-D distances along the edge: s1 = -t*|G|,  s2 = (1-t)*|G|
    s1 = -t_pq * edge_length
    s2 = (1.0 - t_pq) * edge_length

    # ------------------------------------------------------------------ #
    #  Transcendental expressions LN_pq and AN_pq  (Tsoulis Eqs 14-15)
    # ------------------------------------------------------------------ #
    ln_num = jnp.clip(s2 + l2, min=eps)
    ln_den = jnp.clip(s1 + l1, min=eps)
    log_term = jnp.log(ln_num / ln_den)

    h_p_e = jnp.broadcast_to(h_p[:, :, None], h_pq.shape)  # (Q, F, 3)
    degenerate = (h_p_e < 1e-15) | (h_pq < 1e-15)
    arctan_term = jnp.arctan(h_p_e * s2 / jnp.clip(h_pq * l2, min=eps)) - jnp.arctan(
        h_p_e * s1 / jnp.clip(h_pq * l1, min=eps)
    )
    arctan_term = jnp.where(degenerate, jnp.zeros_like(arctan_term), arctan_term)

    # ------------------------------------------------------------------ #
    #  Per-face sums S1, S2, singularity correction  (Tsoulis Eqs 11-12)
    # ------------------------------------------------------------------ #
    sum1 = jnp.sum(h_pq_signed * log_term, axis=-1)  # (Q, F)
    sum2 = jnp.sum(sigma_pq * arctan_term, axis=-1)  # (Q, F)
    sum1_tensor = jnp.sum(n_pq * log_term[..., None], axis=-2)  # (Q, F, 3)

    n_pos = jnp.sum(sigma_pq > 0, axis=-1)  # (Q, F)
    n_zero = jnp.sum(sigma_pq == 0, axis=-1)  # (Q, F)

    all_inside = n_pos == 3
    on_edge = (n_pos == 2) & (n_zero == 1)
    on_vertex = (n_pos == 1) & (n_zero == 2)

    edge_unit_vector = edges / jnp.clip(edge_length[..., None], min=eps)  # (Q, F, 3, 3)
    cos_t0 = -jnp.sum(edge_unit_vector[..., 0, :] * edge_unit_vector[..., 2, :], axis=-1)
    cos_t1 = -jnp.sum(edge_unit_vector[..., 1, :] * edge_unit_vector[..., 0, :], axis=-1)
    cos_t2 = -jnp.sum(edge_unit_vector[..., 2, :] * edge_unit_vector[..., 1, :], axis=-1)
    
    theta_0 = jnp.arccos(jnp.clip(cos_t0, min=-1.0, max=1.0))
    theta_1 = jnp.arccos(jnp.clip(cos_t1, min=-1.0, max=1.0))
    theta_2 = jnp.arccos(jnp.clip(cos_t2, min=-1.0, max=1.0))
    
    theta_v = (
        theta_2 * (sigma_pq[..., 0] > 0).astype(h_p.dtype)
        + theta_0 * (sigma_pq[..., 1] > 0).astype(h_p.dtype)
        + theta_1 * (sigma_pq[..., 2] > 0).astype(h_p.dtype)
    )  # (Q, F)

    singularity_scalar = jnp.where(
        all_inside,
        -2.0 * jnp.pi * h_p,
        jnp.where(
            on_edge,
            -jnp.pi * h_p,
            jnp.where(on_vertex, -theta_v * h_p, jnp.zeros_like(h_p)),
        ),
    )

    singularity_vector = jnp.where(
        all_inside[..., None],
        -2.0 * jnp.pi * sigma_p[..., None] * face_normal,
        jnp.where(
            on_edge[..., None],
            -jnp.pi * sigma_p[..., None] * face_normal,
            jnp.where(
                on_vertex[..., None],
                -theta_v[..., None] * sigma_p[..., None] * face_normal,
                jnp.zeros_like(face_normal),
            ),
        ),
    )  # (Q, F, 3)

    face_sum = sum1 + h_p * sum2 + singularity_scalar  # (Q, F)

    # ------------------------------------------------------------------ #
    #  Sum over faces -> potential and acceleration
    # ------------------------------------------------------------------ #
    potential_terms = sigma_p * h_p * face_sum  # (Q, F)
    g_contributions = face_normal * face_sum[..., None]  # (Q, F, 3)

    potential = (gravitational_constant * density / 2.0) * jnp.sum(potential_terms, axis=-1)
    acceleration = -gravitational_constant * density * jnp.sum(g_contributions, axis=-2)

    # ------------------------------------------------------------------ #
    #  Gradient tensor  (Tsoulis Eq. 13)
    # ------------------------------------------------------------------ #
    tensor_subterm = (
        sum1_tensor + face_normal * (sigma_p * sum2)[..., None] + singularity_vector
    )  # (Q, F, 3)

    # diagonal components: Vxx, Vyy, Vzz
    diag = face_normal * tensor_subterm  # (Q, F, 3)
    
    # off-diagonal: Vxy=N[0]*sub[1], Vxz=N[0]*sub[2], Vyz=N[1]*sub[2]
    offdiag_normal = jnp.stack(
        [face_normal[..., 0], face_normal[..., 0], face_normal[..., 1]], axis=-1
    )
    offdiag_subterm = jnp.stack(
        [tensor_subterm[..., 1], tensor_subterm[..., 2], tensor_subterm[..., 2]], axis=-1
    )
    offdiag = offdiag_normal * offdiag_subterm  # (Q, F, 3)

    tensor = (
        gravitational_constant
        * density
        * jnp.sum(jnp.concatenate([diag, offdiag], axis=-1), axis=-2)
    )  # (Q, 6)

    return potential, acceleration, tensor