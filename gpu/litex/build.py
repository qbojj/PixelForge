"""Build script to convert all GPU pipeline components to Verilog.

This module generates Verilog from all Amaranth GPU components for consumption
by the LiteX wrappers.
"""

from pathlib import Path

from amaranth.back import verilog


def generate_all_verilog(output_dir="build/gpu_verilog"):
    """Generate Verilog from all GPU pipeline components.

    Args:
        output_dir: Directory where Verilog files will be written

    Returns:
        Dictionary mapping component names to their generated Verilog file paths
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Import all components
    from gpu.input_assembly.cores import (
        IndexGenerator,
        InputAssembly,
        InputTopologyProcessor,
    )
    from gpu.pixel_shading.cores import DepthStencilTest, SwapchainOutput, Texturing
    from gpu.primitive_assembly.cores import PrimitiveAssembly
    from gpu.rasterizer.cores import PrimitiveClipper
    from gpu.rasterizer.rasterizer import TriangleRasterizer
    from gpu.vertex_shading.cores import VertexShading
    from gpu.vertex_transform.cores import VertexTransform

    components = {
        "IndexGenerator": IndexGenerator,
        "InputTopologyProcessor": InputTopologyProcessor,
        "InputAssembly": InputAssembly,
        "VertexTransform": VertexTransform,
        "VertexShading": VertexShading,
        "PrimitiveAssembly": PrimitiveAssembly,
        "PrimitiveClipper": PrimitiveClipper,
        "TriangleRasterizer": TriangleRasterizer,
        "Texturing": Texturing,
        "DepthStencilTest": DepthStencilTest,
        "SwapchainOutput": SwapchainOutput,
    }

    generated_files = {}

    for name, component_cls in components.items():
        try:
            # Instantiate the component
            component = component_cls()

            # Convert to Verilog
            verilog_text = verilog.convert(
                component,
                name=name,
            )

            # Write to file
            output_file = output_path / f"{name}.v"
            with open(output_file, "w") as f:
                f.write(verilog_text)

            generated_files[name] = str(output_file)
            print(f"✓ Generated {name}.v")

        except Exception as e:
            print(f"✗ Failed to generate {name}: {e}")

    print(f"\nGenerated Verilog files in {output_dir}")
    return generated_files


def generate_verilog(component_name, output_dir="build/gpu_verilog"):
    """Generate Verilog for a single GPU component.

    Args:
        component_name: Name of the component to generate
        output_dir: Directory where Verilog file will be written

    Returns:
        Path to the generated Verilog file
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Component mapping
    component_map = {
        "IndexGenerator": ("gpu.input_assembly.cores", "IndexGenerator"),
        "InputTopologyProcessor": (
            "gpu.input_assembly.cores",
            "InputTopologyProcessor",
        ),
        "InputAssembly": ("gpu.input_assembly.cores", "InputAssembly"),
        "VertexTransform": ("gpu.vertex_transform.cores", "VertexTransform"),
        "VertexShading": ("gpu.vertex_shading.cores", "VertexShading"),
        "PrimitiveAssembly": ("gpu.primitive_assembly.cores", "PrimitiveAssembly"),
        "PrimitiveClipper": ("gpu.rasterizer.cores", "PrimitiveClipper"),
        "TriangleRasterizer": ("gpu.rasterizer.rasterizer", "TriangleRasterizer"),
        "Texturing": ("gpu.pixel_shading.cores", "Texturing"),
        "DepthStencilTest": ("gpu.pixel_shading.cores", "DepthStencilTest"),
        "SwapchainOutput": ("gpu.pixel_shading.cores", "SwapchainOutput"),
    }

    if component_name not in component_map:
        raise ValueError(f"Unknown component: {component_name}")

    module_name, class_name = component_map[component_name]

    try:
        # Dynamic import
        module = __import__(module_name, fromlist=[class_name])
        component_cls = getattr(module, class_name)

        # Instantiate and convert
        component = component_cls()
        verilog_text = verilog.convert(
            component,
            name=component_name,
        )

        # Write to file
        output_file = output_path / f"{component_name}.v"
        with open(output_file, "w") as f:
            f.write(verilog_text)

        print(f"Generated {component_name}.v at {output_file}")
        return str(output_file)

    except Exception as e:
        print(f"Failed to generate {component_name}: {e}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Generate specific component
        component_name = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "build/gpu_verilog"
        generate_verilog(component_name, output_dir)
    else:
        # Generate all components
        generate_all_verilog()
