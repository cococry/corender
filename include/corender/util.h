#pragma once 

#include <stdio.h>
#include <stdlib.h>

#define _CR_BRAND_NAME "corender"  
#define _CR_VERSION "alpha 0.1"


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


void cr_util_log_header(FILE* stream, enum cr_log_level_t lvl);

char* cr_util_log_get_filepath();
