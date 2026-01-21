#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include <stdint.h>
#include <stddef.h>

/* Simple OBJ file loader for triangle meshes */

typedef struct {
    float x, y, z;
} vec3f;

typedef struct {
    float u, v;
} vec2f;

typedef struct {
    int v_idx;   /* Vertex position index */
    int vt_idx;  /* Texture coordinate index (-1 if not present) */
    int vn_idx;  /* Normal index (-1 if not present) */
} face_vertex;

typedef struct {
    vec3f *positions;      /* Array of vertex positions */
    vec3f *normals;        /* Array of vertex normals */
    vec2f *texcoords;      /* Array of texture coordinates */
    face_vertex *faces;    /* Array of face vertices (triangulated) */

    size_t num_positions;
    size_t num_normals;
    size_t num_texcoords;
    size_t num_faces;      /* Number of triangles * 3 */

    size_t capacity_positions;
    size_t capacity_normals;
    size_t capacity_texcoords;
    size_t capacity_faces;
} obj_model;

/* Load OBJ file from path. Returns 0 on success, -1 on failure */
int obj_load(const char *filename, obj_model *model);

/* Free model data */
void obj_free(obj_model *model);

/* Get bounding box of model */
void obj_get_bounds(const obj_model *model, vec3f *min, vec3f *max);

#endif /* OBJ_LOADER_H */
