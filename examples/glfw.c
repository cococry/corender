
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

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

static inline void draw_line(float x0, float y0, float x1, float y1) {
    cr_draw_segment(&ctx, (struct cr_segment_t){
            .p0 = { x0, y0 },
            .p1 = { x1, y1 }
            });
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


    uint32_t n_glfw_exts = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&n_glfw_exts);

    uint32_t n_exts = n_glfw_exts /* + 1 */;

    const char** exts = malloc(sizeof(char*) * n_exts);

    for (uint32_t i = 0; i < n_glfw_exts; i++) {
        exts[i] = glfw_exts[i];
    }

    //exts[n_glfw_exts] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;



    const char* validation_layers[] = {
        "VK_LAYER_KHRONOS_validation"
    };

    struct cr_context_init_info_t info = {
        .enable_validation = true,
        .n_exts = n_exts,
        .exts = exts,
        .enable_gpu_profiler = true,

        .layers = NULL, 
        .n_layers = 0,

        .log_verbose = true,
        .surface_create = _glfw_surface_create,
        .surface_userdata = window
    };

    cr_context_create(&ctx, &info);

    double start_time = glfwGetTime();

#define N_STARS 40000

    float* star_x = malloc(N_STARS * sizeof(float));
    float* star_y = malloc(N_STARS * sizeof(float));

    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window, &fbw, &fbh);

    uint32_t rng = 0x1512125; 

    /* generate random positions once */
    for (int i = 0; i < N_STARS; ++i) {
        star_x[i] = rand01(&rng) * (float)fbw + 50;
        star_y[i] = rand01(&rng) * (float)fbh + 50;
    }

    while (!glfwWindowShouldClose(window)) {
        glfwGetFramebufferSize(window, &fbw, &fbh);
        float t = glfwGetTime() - start_time; 

        /* pulse size from 100 -> 513 */
        float min_size = 30.0f;
        float max_size = 50.0f;
        float pulse01 = 0.5f + 0.5f * sinf(t * 1.7f);
        float size = 50;

        cr_draw_begin(&ctx);


        cr_draw_begin_path(&ctx);


        for(uint i = 0; i < 1; i++) {
            draw_star(star_x[i], star_y[i], 5, 50, 80, t * 0.2);
        }

cr_draw_end_path(&ctx);

        cr_draw_end(&ctx);
        glfwPollEvents();
    }


    cr_context_destroy(&ctx);
    glfwTerminate();
    return 0;
}
