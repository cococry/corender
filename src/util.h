#pragma once 

#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan_core.h>

#define _CR_BRAND_NAME "corender"  
#define _CR_VERSION "alpha 0.1"

#define CR_MAX(a, b) a > b ? a : b
#define CR_MIN(a, b) a < b ? a : b



#define CR_MAX_BINDING_DESC 2
#define CR_MAX_VERT_ATTRS 5
#define CR_MAX_STATIC_BUFS 8
#define CR_MAX_DYNAMIC_BUFS 8

#define CR_MAX_BATCH 3 * 100000
#define CR_INITIAL_BATCH_CAP 4
#define CR_MAX_PENDING_BUFFER_DESTROYS 32

#define CR_INITIAL_INDIRECT_DRAW_CAP 8 

enum cr_log_level_t {
  CR_LL_TRACE = 0,
  CR_LL_WARN,
  CR_LL_ERR,
  CR_LL_FATAL,
  CR_LL_COUNT
};

#define CR_TRACE(logstate, ...)                                                     \
  if (!(logstate).quiet) {                                                          \
    if ((logstate).verbose) {                                                       \
      do {                                                                          \
        cr_util_log_header((logstate).stream, CR_LL_TRACE);                         \
        fprintf((logstate).stream, "%s: %s: ", _SUBSYS_NAME, __func__);             \
        fprintf((logstate).stream, __VA_ARGS__);                                    \
        fprintf((logstate).stream, "\n");                                           \
        if ((logstate).stream != stdout &&                                          \
            (logstate).stream != stderr && !(logstate).quiet) {                     \
          cr_util_log_header(stdout, CR_LL_TRACE);                                  \
          fprintf(stdout, "%s: %s: ", _SUBSYS_NAME, __func__);                      \
          fprintf(stdout, __VA_ARGS__);                                             \
          fprintf(stdout, "\n");                                                    \
        }                                                                           \
      } while (0);                                                                  \
    }                                                                               \
  }

#define CR_WARN(logstate, ...)                                                      \
  if (!(logstate).quiet) {                                                          \
    do {                                                                            \
      cr_util_log_header((logstate).stream, CR_LL_WARN);                            \
      fprintf((logstate).stream, "%s: %s (%s:%d): ",                                \
              _SUBSYS_NAME, __func__, __FILE__, __LINE__);                          \
      fprintf((logstate).stream, __VA_ARGS__);                                      \
      fprintf((logstate).stream, "\n");                                             \
      if ((logstate).stream != stdout &&                                            \
          (logstate).stream != stderr && !(logstate).quiet) {                       \
        cr_util_log_header(stdout, CR_LL_WARN);                                     \
        fprintf(stdout, "%s: %s (%s:%d): ",                                         \
                _SUBSYS_NAME, __func__, __FILE__, __LINE__);                        \
        fprintf(stdout, __VA_ARGS__);                                               \
        fprintf(stdout, "\n");                                                      \
      }                                                                             \
    } while (0);                                                                    \
  }

#define CR_ERROR(logstate, ...)                                                     \
  if (!(logstate).quiet) {                                                          \
    do {                                                                            \
      FILE* _stream = ((logstate).stream == stdout) ? stderr : (logstate).stream;   \
      cr_util_log_header(_stream, CR_LL_ERR);                                       \
      fprintf(_stream, "%s: %s (%s:%d): ",                                          \
              _SUBSYS_NAME, __func__, __FILE__, __LINE__);                          \
      fprintf(_stream, __VA_ARGS__);                                                \
      fprintf(_stream, "\n");                                                       \
      if ((logstate).stream != stdout &&                                            \
          (logstate).stream != stderr && !(logstate).quiet) {                       \
        cr_util_log_header(stderr, CR_LL_ERR);                                      \
        fprintf(stderr, "%s: %s (%s:%d): ",                                         \
                _SUBSYS_NAME, __func__, __FILE__, __LINE__);                        \
        fprintf(stderr, __VA_ARGS__);                                               \
        fprintf(stderr, "\n");                                                      \
      }                                                                             \
    } while (0);                                                                    \
  }

#define CR_FATAL(logstate, ...)                                                     \
  if (!(logstate).quiet) {                                                          \
    do {                                                                            \
      FILE* _stream = ((logstate).stream == stdout) ? stderr : (logstate).stream;   \
      cr_util_log_header(_stream, CR_LL_FATAL);                                     \
      fprintf(_stream, "%s: %s (%s:%d): ",                                          \
              _SUBSYS_NAME, __func__, __FILE__, __LINE__);                          \
      fprintf(_stream, __VA_ARGS__);                                                \
      fprintf(_stream, "\n");                                                       \
      if ((logstate).stream != stdout &&                                            \
          (logstate).stream != stderr && !(logstate).quiet) {                       \
        cr_util_log_header(stderr, CR_LL_FATAL);                                    \
        fprintf(stderr, "%s: %s (%s:%d): ",                                         \
                _SUBSYS_NAME, __func__, __FILE__, __LINE__);                        \
        fprintf(stderr, __VA_ARGS__);                                               \
        fprintf(stderr, "\n");                                                      \
      }                                                                             \
      exit(1);                                                                      \
    } while (0);                                                                    \
  } \

#define _PARAM_CHECK_FAIL()                                               \
  do {                                                                    \
    fprintf(stderr, "corender: Fatal: Did not pass parameter check.");    \
    exit(1);                                                              \
  } while (0);                                                \

#define _VK_CHECK(ctx, expr)                                  \
do {                                                          \
  VkResult _res = (expr);                                     \
  if (_res != VK_SUCCESS) {                                   \
    CR_ERROR(ctx->log, "Vulkan error: %s (%i) - %s failed.",  \
    cr_util_vk_result_to_string(_res), _res, #expr);          \
    goto err;                                                 \
  }                                                           \
} while (0)


void cr_util_log_header(FILE* stream, enum cr_log_level_t lvl);

char* cr_util_log_get_filepath();

char* cr_util_get_state_folder();

unsigned char* cr_util_read_file(const char* filepath, size_t* o_filesize);

void* cr_util_alloc(void* ctx, size_t n, size_t size);

const char* cr_util_vk_result_to_string(VkResult r);
