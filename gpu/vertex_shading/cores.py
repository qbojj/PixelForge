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

    i: stream.Interface
    o: stream.Interface

    material: Signal
    lights: Signal
    ready: Signal

    def __init__(self, num_lights=8):
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
        n = Signal.like(self.i.p.normal_view)
        v_color = Signal.like(self.i.p.color)

        # Single shared multiplier
        mul_a = Signal(FixedPoint)
        mul_b = Signal(FixedPoint)
        mul_result = Signal(FixedPoint)
        m.d.comb += mul_result.eq(mul_a * mul_b)

        # Accumulators for dot product and shading
        dot_accum = Signal(FixedPoint)
        dp_clamped = Signal(FixedPoint)

        amb_accum = Signal(data.ArrayLayout(FixedPoint, 3))
        dif_accum = Signal(data.ArrayLayout(FixedPoint, 3))

        # Output color (accumulated across all lights)
        amb_comp = Signal(FixedPoint)
        dif_comp = Signal(FixedPoint)

        out_comp = Signal(FixedPoint)

        ch_idx = Signal(range(3))

        light_idx = Signal(range(self._num_lights))
        light = Signal(LightPropertyLayout)

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                m.d.comb += self.ready.eq(1)

                with m.If(self.i.valid):
                    # Capture input
                    m.d.sync += n.eq(self.i.p.normal_view)
                    m.d.sync += v_color.eq(self.i.p.color)
                    m.d.sync += self.o.p.position_ndc.eq(self.i.p.position_proj)
                    m.d.sync += self.o.p.texcoords.eq(self.i.p.texcoords)
                    m.d.sync += self.o.p.color[3].eq(self.i.p.color[3])
                    m.d.sync += light_idx.eq(0)
                    m.d.sync += amb_accum.eq(0)
                    m.d.sync += dif_accum.eq(0)
                    m.next = "LIGHT_START"

            with m.State("LIGHT_START"):
                m.d.sync += light.eq(Array(self.lights)[light_idx])
                m.next = "DOT_0"

            with m.State("DOT_0"):
                m.d.comb += [
                    mul_a.eq(n[0]),
                    mul_b.eq(-light.position[0]),
                ]
                m.d.sync += dot_accum.eq(mul_result)
                m.next = "DOT_1"

            with m.State("DOT_1"):
                m.d.comb += [
                    mul_a.eq(n[1]),
                    mul_b.eq(-light.position[1]),
                ]
                m.d.sync += dot_accum.eq(dot_accum + mul_result)
                m.next = "DOT_2"

            with m.State("DOT_2"):
                m.d.comb += [
                    mul_a.eq(n[2]),
                    mul_b.eq(-light.position[2]),
                ]
                m.d.sync += dot_accum.eq(dot_accum + mul_result)
                m.d.sync += dp_clamped.eq(
                    Mux(dot_accum + mul_result > 0, dot_accum + mul_result, 0)
                )
                m.d.sync += ch_idx.eq(0)
                m.next = "GET_CHANNEL"

            with m.State("GET_CHANNEL"):
                m.d.sync += amb_accum[ch_idx].eq(
                    amb_accum[ch_idx] + light.ambient[ch_idx]
                )
                m.d.comb += [
                    mul_a.eq(light.diffuse[ch_idx]),
                    mul_b.eq(dp_clamped),
                ]
                m.d.sync += dif_accum[ch_idx].eq(dif_accum[ch_idx] + mul_result)

                m.d.sync += ch_idx.eq(ch_idx + 1)
                with m.If(ch_idx == 3):
                    with m.If(light_idx + 1 == self._num_lights):
                        m.d.sync += ch_idx.eq(0)
                        m.next = "FINALIZE_AMBIENT"
                    with m.Else():
                        m.d.sync += light_idx.eq(light_idx + 1)
                        m.next = "LIGHT_START"

            with m.State("FINALIZE_AMBIENT"):
                m.d.comb += [
                    mul_a.eq(v_color[ch_idx]),
                    mul_b.eq(self.material.ambient[ch_idx]),
                ]
                m.d.sync += amb_comp.eq(mul_result)
                m.next = "MODULATE_AMBIENT"

            with m.State("MODULATE_AMBIENT"):
                m.d.comb += [
                    mul_a.eq(amb_accum[ch_idx]),
                    mul_b.eq(amb_comp),
                ]
                m.d.sync += out_comp.eq(mul_result)
                m.next = "FINALIZE_DIFFUSE"

            with m.State("FINALIZE_DIFFUSE"):
                m.d.comb += [
                    mul_a.eq(v_color[ch_idx]),
                    mul_b.eq(self.material.diffuse[ch_idx]),
                ]
                m.d.sync += dif_comp.eq(mul_result)
                m.next = "MODULATE_DIFFUSE"

            with m.State("MODULATE_DIFFUSE"):
                m.d.comb += [
                    mul_a.eq(dif_accum[ch_idx]),
                    mul_b.eq(dif_comp),
                ]
                m.d.sync += out_comp.eq(out_comp + mul_result)
                m.next = "SAVE"

            with m.State("SAVE"):
                m.d.sync += self.o.p.color[ch_idx].eq(out_comp)
                m.d.sync += ch_idx.eq(ch_idx + 1)
                with m.If(ch_idx == 3):
                    m.next = "SEND"
                with m.Else():
                    m.next = "FINALIZE_AMBIENT"

            with m.State("SEND"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
