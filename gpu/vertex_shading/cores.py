from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.wiring import In, Out

from ..utils.layouts import PrimitiveAssemblyLayout, ShadingVertexLayout
from ..utils.types import FixedPoint, Vector3_mem, Vector4_mem


class LightPropertyLayout(data.Struct):
    """Light properties layout"""

    position: Vector4_mem
    ambient: Vector3_mem
    diffuse: Vector3_mem
    specular: Vector3_mem


class MaterialPropertyLayout(data.Struct):
    """Material properties layout"""

    ambient: Vector3_mem
    diffuse: Vector3_mem
    specular: Vector3_mem
    shininess: unsigned(32)


class VertexShading(wiring.Component):
    """Vertex shading core

    Shades incoming vertices using Gouraud shading model.
    Outputs shaded vertices for rasterization stage.

    Input: ShadingVertexLayout
    Output: ShadingVertexLayout

    Uses following wires for material properties:
    - material_ambient: Ambient color of the material (vec3)
    - material_diffuse: Diffuse color of the material (vec3)
    - material_specular: Specular color of the material (vec3)
    - material_shininess: Shininess coefficient of the material (float)

    Uses following wires for light properties:
    - light: array of light property structures

    TODO: for now only directional lights are supported
    """

    def __init__(self, num_lights=1):
        self._num_lights = num_lights
        super().__init__(
            {
                "i": In(stream.Signature(ShadingVertexLayout)),
                "o": Out(stream.Signature(PrimitiveAssemblyLayout)),
                "material": In(MaterialPropertyLayout),
                "lights": In(LightPropertyLayout).array(num_lights),
                "ready": Out(1),
            }
        )

    def elaborate(self, platform):
        m = Module()

        # Cached vertex and light data
        n = Array(Signal(FixedPoint) for _ in range(3))
        v_color = Array(Signal.like(self.i.p.color[i]) for i in range(4))
        v_pos_ndc = Signal.like(self.i.p.position_proj)
        v_texcoords = Signal.like(self.i.p.texcoords)

        # Single shared multiplier
        mul_a = Signal(FixedPoint)
        mul_b = Signal(FixedPoint)
        mul_result = Signal(FixedPoint)
        m.d.comb += mul_result.eq(mul_a * mul_b)

        # Accumulators for dot product and shading
        dot_accum = Signal(FixedPoint)
        dp_clamped = Signal(FixedPoint)
        amb_accum = Signal(FixedPoint)
        dif_accum = Signal(FixedPoint)

        # Output color (accumulated across all lights)
        out_color = Signal.like(self.o.p.color)

        with m.If(self.o.ready):
            m.d.sync += self.o.valid.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                m.d.comb += self.ready.eq(1)

                with m.If(self.i.valid):
                    # Capture input
                    m.d.sync += [
                        n[0].eq(self.i.p.normal_view[0]),
                        n[1].eq(self.i.p.normal_view[1]),
                        n[2].eq(self.i.p.normal_view[2]),
                    ] + [v_color[i].eq(self.i.p.color[i]) for i in range(4)]
                    m.d.sync += [
                        v_pos_ndc.eq(self.i.p.position_proj),
                        v_texcoords.eq(self.i.p.texcoords),
                        out_color.eq(0),  # Initialize accumulated color
                    ]
                    m.d.sync += dot_accum.eq(0)
                    m.d.sync += Print("Shading vtx in: ", self.i.p)
                    m.next = "DOT_0_LIGHT_0"

            # Nested loops: for each light, compute dot product and per-channel shading
            for light_idx in range(self._num_lights):
                # Dot product computation for this light (3 cycles)
                with m.State(f"DOT_0_LIGHT_{light_idx}"):
                    m.d.comb += [
                        mul_a.eq(n[0]),
                        mul_b.eq(-self.lights[light_idx].position[0]),
                    ]
                    m.d.sync += dot_accum.eq(mul_result)
                    m.next = f"DOT_1_LIGHT_{light_idx}"

                with m.State(f"DOT_1_LIGHT_{light_idx}"):
                    m.d.comb += [
                        mul_a.eq(n[1]),
                        mul_b.eq(-self.lights[light_idx].position[1]),
                    ]
                    m.d.sync += dot_accum.eq(dot_accum + mul_result)
                    m.next = f"DOT_2_LIGHT_{light_idx}"

                with m.State(f"DOT_2_LIGHT_{light_idx}"):
                    m.d.comb += [
                        mul_a.eq(n[2]),
                        mul_b.eq(-self.lights[light_idx].position[2]),
                    ]
                    m.d.sync += dot_accum.eq(dot_accum + mul_result)
                    m.d.sync += dp_clamped.eq(
                        Mux(dot_accum + mul_result > 0, dot_accum + mul_result, 0)
                    )
                    m.next = f"COLOR_AMBIENT_0_LIGHT_{light_idx}"

                # Per-channel shading for this light
                for ch_idx in range(3):
                    next_ch = ch_idx + 1
                    if next_ch < 3:
                        next_state_name = f"COLOR_AMBIENT_{next_ch}_LIGHT_{light_idx}"
                    else:
                        # Last channel of this light
                        if light_idx + 1 < self._num_lights:
                            # Move to next light
                            next_state_name = f"DOT_0_LIGHT_{light_idx + 1}"
                        else:
                            # Last light, last channel -> go to MODULATE
                            next_state_name = "MODULATE_BY_VERTEX_COLOR_0"

                    with m.State(f"COLOR_AMBIENT_{ch_idx}_LIGHT_{light_idx}"):
                        m.d.comb += [
                            mul_a.eq(self.material.ambient[ch_idx]),
                            mul_b.eq(self.lights[light_idx].ambient[ch_idx]),
                        ]
                        m.d.sync += amb_accum.eq(mul_result)
                        m.next = f"COLOR_DIFFUSE_{ch_idx}_LIGHT_{light_idx}"

                    with m.State(f"COLOR_DIFFUSE_{ch_idx}_LIGHT_{light_idx}"):
                        m.d.comb += [
                            mul_a.eq(self.material.diffuse[ch_idx]),
                            mul_b.eq(self.lights[light_idx].diffuse[ch_idx]),
                        ]
                        m.d.sync += dif_accum.eq(mul_result)
                        m.next = f"COLOR_DIFFUSE_MUL_{ch_idx}_LIGHT_{light_idx}"

                    with m.State(f"COLOR_DIFFUSE_MUL_{ch_idx}_LIGHT_{light_idx}"):
                        m.d.comb += [
                            mul_a.eq(dif_accum),
                            mul_b.eq(dp_clamped),
                        ]
                        m.d.sync += dif_accum.eq(mul_result)
                        m.next = f"COLOR_ACCUMULATE_{ch_idx}_LIGHT_{light_idx}"

                    with m.State(f"COLOR_ACCUMULATE_{ch_idx}_LIGHT_{light_idx}"):
                        # Accumulate light contribution (add to out_color)
                        m.d.sync += out_color[ch_idx].eq(
                            out_color[ch_idx] + amb_accum + dif_accum
                        )
                        m.next = next_state_name

            # Final modulation by vertex color (per-channel)
            for ch_idx in range(3):
                next_ch = ch_idx + 1
                next_state_name = (
                    f"MODULATE_BY_VERTEX_COLOR_{next_ch}" if next_ch < 3 else "SEND"
                )

                with m.State(f"MODULATE_BY_VERTEX_COLOR_{ch_idx}"):
                    m.d.comb += [
                        mul_a.eq(v_color[ch_idx]),
                        mul_b.eq(out_color[ch_idx]),
                    ]
                    m.d.sync += out_color[ch_idx].eq(
                        mul_result.saturate(v_color[ch_idx].shape())
                    )
                    if ch_idx == 2:
                        # Last channel: preserve alpha
                        m.d.sync += out_color[3].eq(v_color[3])
                    m.next = next_state_name

            with m.State("SEND"):
                with m.If(~self.o.valid | self.o.ready):
                    m.d.sync += [
                        self.o.p.position_ndc.eq(v_pos_ndc),
                        self.o.p.texcoords.eq(v_texcoords),
                        self.o.p.color.eq(out_color),
                        self.o.valid.eq(1),
                    ]
                    m.next = "IDLE"

        return m
