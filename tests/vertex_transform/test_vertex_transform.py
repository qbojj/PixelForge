import numpy as np
import pytest
from amaranth import *
from amaranth.sim import Simulator
from numpy.linalg import inv

from gpu.utils.layouts import num_textures
from gpu.vertex_transform.cores import VertexTransform

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


def identity_mat(size):
    return np.identity(size)


def make_vertex():
    return {
        "position": [1.0, -2.0, 3.0, 1.0],
        "normal": [0.0, 0.0, 1.0],
        "texcoords": [[0.1, 0.2, 0.3, 1.0] for _ in range(num_textures)],
        "color": [0.25, 0.5, 0.75, 1.0],
    }


def test_identity_transform_positions():
    dut = VertexTransform()
    t = SimpleTestbench(dut)

    mv = identity_mat(4)
    proj = identity_mat(4)
    vertex = make_vertex()

    mv_inv_t = inv(mv[:3, :3]).T

    async def init_proc(ctx):
        # Set transformation matrices
        ctx.set(dut.position_mv, mv.flatten().tolist())
        ctx.set(dut.position_p, proj.flatten().tolist())
        ctx.set(dut.normal_mv_inv_t, mv_inv_t.flatten().tolist())

        # Disable normal and texture transforms
        ctx.set(dut.enabled.normal, 0)
        for i in range(num_textures):
            ctx.set(dut.enabled.texture[i], 0)

        # Set texture transforms to identity
        for i in range(num_textures):
            ctx.set(dut.texture_transforms[i], identity_mat(4).flatten().tolist())

    async def output_checker(ctx, results):
        assert len(results) == 1
        out = results[0]

        def vec_to_list(vec):
            return [c.as_float() for c in vec]

        assert vec_to_list(out.position_view) == pytest.approx(vertex["position"])
        assert vec_to_list(out.position_proj) == pytest.approx(vertex["position"])

        # Disabled normal/tex transforms should zero normals and leave texcoords as identity defaults (0,0,0,1)
        assert vec_to_list(out.normal_view) == pytest.approx([0.0, 0.0, 0.0])
        for tex_idx in range(num_textures):
            assert vec_to_list(out.texcoords[tex_idx]) == pytest.approx(
                [0.0, 0.0, 0.0, 1.0]
            )

    sim = Simulator(t)
    sim.add_clock(1e-6)
    stream_testbench(
        sim,
        input_stream=dut.i,
        input_data=[vertex],
        output_stream=dut.o,
        output_data_checker=output_checker,
        init_process=init_proc,
        is_finished=dut.ready,
    )

    sim.run()
