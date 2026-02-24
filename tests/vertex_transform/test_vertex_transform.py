import numpy as np
import pytest
from amaranth import *
from amaranth.sim import Simulator

from gpu.utils.layouts import num_textures
from gpu.vertex_transform.cores import VertexTransform

from ..utils.streams import stream_testbench
from ..utils.testbench import SimpleTestbench


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

    mv = np.identity(4)
    proj = np.identity(4)
    mv_inv_t = np.identity(3)
    tex = np.identity(4)

    vertex = make_vertex()

    async def init_proc(ctx):
        # Set transformation matrices
        ctx.set(dut.position_mv, mv.flatten().tolist())
        ctx.set(dut.position_p, proj.flatten().tolist())
        ctx.set(dut.normal_mv_inv_t, mv_inv_t.flatten().tolist())
        ctx.set(dut.texture_transform, tex.flatten().tolist())

    async def output_checker(ctx, results):
        assert len(results) == 1
        out = results[0]

        def vec_to_list(vec):
            return [c.as_float() for c in vec]

        pv = vec_to_list(out.position_view)
        pp = vec_to_list(out.position_proj)
        nv = vec_to_list(out.normal_view)
        tex_vals = [vec_to_list(v) for v in out.texcoords]
        color = vec_to_list(out.color)

        print(
            {
                "pv": pv,
                "pp": pp,
                "nv": nv,
            }
        )

        assert pv == pytest.approx(vertex["position"])
        assert pp == pytest.approx(vertex["position"])
        assert nv == pytest.approx(vertex["normal"])

        for tex_idx in range(num_textures):
            assert tex_vals[tex_idx] == pytest.approx(vertex["texcoords"][tex_idx])

        assert color == pytest.approx(vertex["color"])

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
