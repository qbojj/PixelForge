"""Stream payload layouts for LiteX GPU wrappers."""

# Stream layouts as list of (name, width) tuples for Endpoint
# Use generic "data" field for all payloads to allow .connect() between stages
index_layout = [("data", 32)]
vertex_layout = [("data", 128)]
primitive_layout = [("data", 128)]
fragment_layout = [("data", 128)]

__all__ = [
    "index_layout",
    "vertex_layout",
    "primitive_layout",
    "fragment_layout",
]
