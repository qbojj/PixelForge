#include "obj_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define INITIAL_CAPACITY 1024

static void ensure_capacity_positions(obj_model *model) {
    if (model->num_positions >= model->capacity_positions) {
        model->capacity_positions = model->capacity_positions ? model->capacity_positions * 2 : INITIAL_CAPACITY;
        model->positions = realloc(model->positions, model->capacity_positions * sizeof(vec3f));
    }
}

static void ensure_capacity_normals(obj_model *model) {
    if (model->num_normals >= model->capacity_normals) {
        model->capacity_normals = model->capacity_normals ? model->capacity_normals * 2 : INITIAL_CAPACITY;
        model->normals = realloc(model->normals, model->capacity_normals * sizeof(vec3f));
    }
}

static void ensure_capacity_texcoords(obj_model *model) {
    if (model->num_texcoords >= model->capacity_texcoords) {
        model->capacity_texcoords = model->capacity_texcoords ? model->capacity_texcoords * 2 : INITIAL_CAPACITY;
        model->texcoords = realloc(model->texcoords, model->capacity_texcoords * sizeof(vec2f));
    }
}

static void ensure_capacity_faces(obj_model *model, size_t count) {
    while (model->num_faces + count > model->capacity_faces) {
        model->capacity_faces = model->capacity_faces ? model->capacity_faces * 2 : INITIAL_CAPACITY;
        model->faces = realloc(model->faces, model->capacity_faces * sizeof(face_vertex));
    }
}

int obj_load(const char *filename, obj_model *model) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Failed to open OBJ file: %s\n", filename);
        return -1;
    }

    memset(model, 0, sizeof(obj_model));

    char line[1024];
    int line_num = 0;

    while (fgets(line, sizeof(line), f)) {
        line_num++;

        /* Skip empty lines and comments */
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        /* Vertex position */
        if (line[0] == 'v' && line[1] == ' ') {
            ensure_capacity_positions(model);
            vec3f *v = &model->positions[model->num_positions++];
            if (sscanf(line + 2, "%f %f %f", &v->x, &v->y, &v->z) != 3) {
                fprintf(stderr, "Invalid vertex at line %d\n", line_num);
            }
        }
        /* Vertex normal */
        else if (line[0] == 'v' && line[1] == 'n' && line[2] == ' ') {
            ensure_capacity_normals(model);
            vec3f *vn = &model->normals[model->num_normals++];
            if (sscanf(line + 3, "%f %f %f", &vn->x, &vn->y, &vn->z) != 3) {
                fprintf(stderr, "Invalid normal at line %d\n", line_num);
            }
        }
        /* Texture coordinate */
        else if (line[0] == 'v' && line[1] == 't' && line[2] == ' ') {
            ensure_capacity_texcoords(model);
            vec2f *vt = &model->texcoords[model->num_texcoords++];
            if (sscanf(line + 3, "%f %f", &vt->u, &vt->v) != 2) {
                fprintf(stderr, "Invalid texcoord at line %d\n", line_num);
            }
        }
        /* Face */
        else if (line[0] == 'f' && line[1] == ' ') {
            face_vertex face_verts[32];
            int vert_count = 0;
            char *p = line + 2;

            /* Parse all vertices in the face */
            while (*p && vert_count < 32) {
                while (isspace(*p)) p++;
                if (!*p) break;

                face_vertex *fv = &face_verts[vert_count];
                fv->v_idx = fv->vt_idx = fv->vn_idx = -1;

                /* Parse vertex/texcoord/normal indices */
                char *start = p;
                fv->v_idx = strtol(p, &p, 10) - 1;  /* OBJ indices are 1-based */

                if (*p == '/') {
                    p++;
                    if (*p != '/') {
                        fv->vt_idx = strtol(p, &p, 10) - 1;
                    }
                    if (*p == '/') {
                        p++;
                        fv->vn_idx = strtol(p, &p, 10) - 1;
                    }
                }

                if (p == start) break;  /* No progress, stop parsing */
                vert_count++;
            }

            /* Triangulate polygon (simple fan triangulation) */
            if (vert_count >= 3) {
                ensure_capacity_faces(model, (vert_count - 2) * 3);
                for (int i = 1; i < vert_count - 1; i++) {
                    model->faces[model->num_faces++] = face_verts[0];
                    model->faces[model->num_faces++] = face_verts[i];
                    model->faces[model->num_faces++] = face_verts[i + 1];
                }
            }
        }
    }

    fclose(f);

    printf("Loaded OBJ: %zu vertices, %zu normals, %zu texcoords, %zu triangles\n",
           model->num_positions, model->num_normals, model->num_texcoords, model->num_faces / 3);

    return 0;
}

void obj_free(obj_model *model) {
    if (model) {
        free(model->positions);
        free(model->normals);
        free(model->texcoords);
        free(model->faces);
        memset(model, 0, sizeof(obj_model));
    }
}

void obj_get_bounds(const obj_model *model, vec3f *min, vec3f *max) {
    if (!model || model->num_positions == 0) {
        min->x = min->y = min->z = 0.0f;
        max->x = max->y = max->z = 0.0f;
        return;
    }

    *min = *max = model->positions[0];

    for (size_t i = 1; i < model->num_positions; i++) {
        const vec3f *v = &model->positions[i];
        if (v->x < min->x) min->x = v->x;
        if (v->y < min->y) min->y = v->y;
        if (v->z < min->z) min->z = v->z;
        if (v->x > max->x) max->x = v->x;
        if (v->y > max->y) max->y = v->y;
        if (v->z > max->z) max->z = v->z;
    }
}
