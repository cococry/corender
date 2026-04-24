#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../include/corender/corender.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static struct cr_context_t ctx;

static void _glfw_resize(GLFWwindow* window, int width, int height) {
    (void)window;
    cr_surface_resize(&ctx, width, height);
}

static bool _glfw_surface_create(
        VkInstance instance,
        struct cr_surface_t* o_surf,
        void* userdata) {
    GLFWwindow* win = (GLFWwindow*)userdata;

    glfwCreateWindowSurface(instance, win, NULL, &o_surf->surf);

    if (o_surf->surf == NULL) {
        printf("Error: failed to create window surface.\n");
    }

    int w, h;
    glfwGetFramebufferSize(win, &w, &h);

    o_surf->width = (uint32_t)w;
    o_surf->height = (uint32_t)h;

    return true;
}

struct cr_point_t {
    float x, y;
};

static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline float rand01(uint32_t* state) {
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f);
}

static inline float frand_range(uint32_t* state, float min_v, float max_v) {
    return min_v + (max_v - min_v) * rand01(state);
}

static inline void draw_line(float x0, float y0, float x1, float y1) {
    cr_draw_segment(&ctx, (struct cr_segment_t){
            .p0 = { x0, y0 },
            .p1 = { x1, y1 }
            });
}

static inline struct cr_point_t rotate_point(struct cr_point_t p, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    struct cr_point_t out = {
        p.x * c - p.y * s,
        p.x * s + p.y * c
    };
    return out;
}

static inline struct cr_point_t transform_point(
        float cx, float cy,
        float x, float y,
        float angle,
        float scale_x,
        float scale_y) {
    struct cr_point_t p = { x * scale_x, y * scale_y };
    p = rotate_point(p, angle);
    p.x += cx;
    p.y += cy;
    return p;
}

static void draw_rect_outline(float x, float y, float w, float h) {
    draw_line(x,     y,     x + w, y);
    draw_line(x + w, y,     x + w, y + h);
    draw_line(x + w, y + h, x,     y + h);
    draw_line(x,     y + h, x,     y);
}

static void draw_triangle_outline(
        float cx, float cy,
        float size,
        float angle) {
    struct cr_point_t p0 = transform_point(cx, cy,  0.0f,     -size, angle, 1.0f, 1.0f);
    struct cr_point_t p1 = transform_point(cx, cy, -0.866f*size, 0.5f*size, angle, 1.0f, 1.0f);
    struct cr_point_t p2 = transform_point(cx, cy,  0.866f*size, 0.5f*size, angle, 1.0f, 1.0f);

    draw_line(p0.x, p0.y, p1.x, p1.y);
    draw_line(p1.x, p1.y, p2.x, p2.y);
    draw_line(p2.x, p2.y, p0.x, p0.y);
}

static void draw_regular_polygon(
        float cx, float cy,
        int sides,
        float radius,
        float angle) {
    if (sides < 3) return;

    for (int i = 0; i < sides; ++i) {
        float a0 = angle + (2.0f * (float)M_PI * (float)i) / (float)sides;
        float a1 = angle + (2.0f * (float)M_PI * (float)(i + 1)) / (float)sides;

        float x0 = cx + cosf(a0) * radius;
        float y0 = cy + sinf(a0) * radius;
        float x1 = cx + cosf(a1) * radius;
        float y1 = cy + sinf(a1) * radius;

        draw_line(x0, y0, x1, y1);
    }
}

static void draw_star(
        float cx, float cy,
        int points,
        float inner_r,
        float outer_r,
        float angle) {
    if (points < 2) return;

    int n = points * 2;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        float r0 = (i & 1) ? inner_r : outer_r;
        float r1 = (j & 1) ? inner_r : outer_r;

        float a0 = angle + ((2.0f * (float)M_PI) * (float)i) / (float)n;
        float a1 = angle + ((2.0f * (float)M_PI) * (float)j) / (float)n;

        float x0 = cx + cosf(a0) * r0;
        float y0 = cy + sinf(a0) * r0;
        float x1 = cx + cosf(a1) * r1;
        float y1 = cy + sinf(a1) * r1;

        draw_line(x0, y0, x1, y1);
    }
}

static void draw_cross(float cx, float cy, float size, float angle) {
    struct cr_point_t a0 = transform_point(cx, cy, -size, 0.0f, angle, 1.0f, 1.0f);
    struct cr_point_t a1 = transform_point(cx, cy,  size, 0.0f, angle, 1.0f, 1.0f);
    struct cr_point_t b0 = transform_point(cx, cy, 0.0f, -size, angle, 1.0f, 1.0f);
    struct cr_point_t b1 = transform_point(cx, cy, 0.0f,  size, angle, 1.0f, 1.0f);

    draw_line(a0.x, a0.y, a1.x, a1.y);
    draw_line(b0.x, b0.y, b1.x, b1.y);
}

static void draw_x_shape(float cx, float cy, float size, float angle) {
    struct cr_point_t a0 = transform_point(cx, cy, -size, -size, angle, 1.0f, 1.0f);
    struct cr_point_t a1 = transform_point(cx, cy,  size,  size, angle, 1.0f, 1.0f);
    struct cr_point_t b0 = transform_point(cx, cy,  size, -size, angle, 1.0f, 1.0f);
    struct cr_point_t b1 = transform_point(cx, cy, -size,  size, angle, 1.0f, 1.0f);

    draw_line(a0.x, a0.y, a1.x, a1.y);
    draw_line(b0.x, b0.y, b1.x, b1.y);
}

static void draw_arc_polyline(
        float cx, float cy,
        float radius_x,
        float radius_y,
        float angle_offset,
        float angle_span,
        int segments) {
    if (segments < 1) return;

    for (int i = 0; i < segments; ++i) {
        float t0 = (float)i / (float)segments;
        float t1 = (float)(i + 1) / (float)segments;

        float a0 = angle_offset + t0 * angle_span;
        float a1 = angle_offset + t1 * angle_span;

        float x0 = cx + cosf(a0) * radius_x;
        float y0 = cy + sinf(a0) * radius_y;
        float x1 = cx + cosf(a1) * radius_x;
        float y1 = cy + sinf(a1) * radius_y;

        draw_line(x0, y0, x1, y1);
    }
}

static void draw_spiral(
        float cx, float cy,
        float start_r,
        float end_r,
        float turns,
        float angle_offset,
        int segments) {
    if (segments < 1) return;

    for (int i = 0; i < segments; ++i) {
        float t0 = (float)i / (float)segments;
        float t1 = (float)(i + 1) / (float)segments;

        float r0 = start_r + (end_r - start_r) * t0;
        float r1 = start_r + (end_r - start_r) * t1;

        float a0 = angle_offset + t0 * turns * 2.0f * (float)M_PI;
        float a1 = angle_offset + t1 * turns * 2.0f * (float)M_PI;

        float x0 = cx + cosf(a0) * r0;
        float y0 = cy + sinf(a0) * r0;
        float x1 = cx + cosf(a1) * r1;
        float y1 = cy + sinf(a1) * r1;

        draw_line(x0, y0, x1, y1);
    }
}

static void draw_zigzag(
        float x,
        float y,
        float w,
        float h,
        int teeth) {
    if (teeth < 1) return;

    float step = w / (float)teeth;
    float px = x;
    float py = y + h * 0.5f;

    for (int i = 1; i <= teeth; ++i) {
        float nx = x + step * (float)i;
        float ny = (i & 1) ? y : (y + h);
        draw_line(px, py, nx, ny);
        px = nx;
        py = ny;
    }
}

static void draw_grid_region(
        float x,
        float y,
        float w,
        float h,
        int cols,
        int rows,
        float skew_x,
        float skew_y) {
    draw_rect_outline(x, y, w, h);

    for (int i = 1; i < cols; ++i) {
        float t = (float)i / (float)cols;
        float xx = x + w * t;
        draw_line(xx, y, xx + skew_x, y + h);
    }

    for (int j = 1; j < rows; ++j) {
        float t = (float)j / (float)rows;
        float yy = y + h * t;
        draw_line(x, yy, x + w, yy + skew_y);
    }
}

static void draw_radial_burst(
        float cx,
        float cy,
        float radius0,
        float radius1,
        int count,
        float angle_offset) {
    if (count < 1) return;

    for (int i = 0; i < count; ++i) {
        float a = angle_offset + 2.0f * (float)M_PI * ((float)i / (float)count);
        float x0 = cx + cosf(a) * radius0;
        float y0 = cy + sinf(a) * radius0;
        float x1 = cx + cosf(a) * radius1;
        float y1 = cy + sinf(a) * radius1;
        draw_line(x0, y0, x1, y1);
    }
}

static void draw_random_segment_cloud(
        uint32_t* rng_state,
        float x,
        float y,
        float w,
        float h,
        int count) {
    for (int i = 0; i < count; ++i) {
        float x0 = frand_range(rng_state, x, x + w);
        float y0 = frand_range(rng_state, y, y + h);
        float x1 = frand_range(rng_state, x, x + w);
        float y1 = frand_range(rng_state, y, y + h);
        draw_line(x0, y0, x1, y1);
    }
}

static void draw_lissajous(
        float cx,
        float cy,
        float ax,
        float ay,
        float fx,
        float fy,
        float phase,
        int segments) {
    if (segments < 2) return;

    float prev_x = cx + sinf(phase) * ax;
    float prev_y = cy + sinf(0.0f) * ay;

    for (int i = 1; i <= segments; ++i) {
        float t = (2.0f * (float)M_PI * (float)i) / (float)segments;
        float x = cx + sinf(fx * t + phase) * ax;
        float y = cy + sinf(fy * t) * ay;
        draw_line(prev_x, prev_y, x, y);
        prev_x = x;
        prev_y = y;
    }
}

int main(void) {
    GLFWwindow* window;

    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(1280, 720, "corender - sophisticated line segment test", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, _glfw_resize);

    uint32_t n_exts = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&n_exts);



    const char* validation_layers[] = {
        "VK_LAYER_KHRONOS_validation"
    };

    struct cr_context_init_info_t info = {
        .enable_validation = true,
        .n_exts = n_exts,
        .exts = exts,
        .enable_time_measuring = false,

        .layers = validation_layers,
        .n_layers = 1,

        .log_verbose = true,
        .surface_create = _glfw_surface_create,
        .surface_userdata = window
    };

    cr_context_create(&ctx, &info);

    double start_time = glfwGetTime();

#define N_STARS 200 

    float star_x[N_STARS];
    float star_y[N_STARS];

    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window, &fbw, &fbh);

    uint32_t rng = time(NULL); 

    /* generate random positions once */
    for (int i = 0; i < N_STARS; ++i) {
        star_x[i] = rand01(&rng) * (float)fbw + 50;
        star_y[i] = rand01(&rng) * (float)fbh + 50;
    }

    while (!glfwWindowShouldClose(window)) {
        glfwGetFramebufferSize(window, &fbw, &fbh);
        float t = (float)(glfwGetTime() - start_time);

        /* pulse size from 100 -> 513 */
        float min_size = 30.0f;
        float max_size = 50.0f;
        float pulse01 = 0.5f + 0.5f * sinf(t * 1.7f);
        float size = min_size + (max_size - min_size) * pulse01;

        cr_draw_begin(&ctx);

        for (int i = 0; i < N_STARS; ++i) {
            draw_star(
                    star_x[i],
                    star_y[i],
                    5,
                    size * 0.45f,   /* inner radius */
                    size,           /* outer radius */
                    t * 0.8f
                    );
        }

        cr_draw_end(&ctx);
        glfwPollEvents();
    }


    cr_context_destroy(&ctx);
    glfwTerminate();
    return 0;
}
