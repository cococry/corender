
#define _GNU_SOURCE
#include <stdbool.h>
#include <stdio.h>
#include "util.h"
#include "../include/corender/corender.h"
#include <time.h>
#include <stdlib.h>
#include <linux/limits.h>
#include <unistd.h>
#include <sys/stat.h>
#include <unistd.h>

#define _SUBSYS_NAME "UTIL"

struct cr_gpu_profiler_t global_profiler;

char* cr_util_get_state_folder(void) {
  const char* state_home = getenv("XDG_STATE_HOME");
  const char* home = getenv("HOME");

  if (!state_home && !home)
    return NULL;

  char base[PATH_MAX];

  if (state_home) {
    snprintf(base, sizeof(base), "%s", state_home);
  } else {
    snprintf(base, sizeof(base), "%s/.local/state", home);
  }

  mkdir(base, 0755); 
  char* dir = malloc(PATH_MAX);
  if (!dir)
    return NULL;

  snprintf(dir, PATH_MAX, "%s", base); 

  mkdir(dir, 0755);

  return dir;
}
char* 
cr_util_log_get_filepath() {
  time_t now = time(NULL);
  struct tm t;
  pid_t pid = getpid();

  char* state_dir = cr_util_get_state_folder();
  char log_dir[PATH_MAX];
  snprintf(log_dir, sizeof(log_dir), "%s/%s/logs", state_dir, _CR_BRAND_NAME);

  // create directories if they don't exist
  char app_dir[PATH_MAX];
  snprintf(app_dir, sizeof(app_dir), "%s/%s", state_dir, _CR_BRAND_NAME);
  mkdir(app_dir, 0755);
  mkdir(log_dir, 0755);

  localtime_r(&now, &t);
  char timestamp[32];
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%d-%H%M", &t);

  // scheme: <log_dir>/<appname>-<timestamp>-<pid>.log
  static char logfile[sizeof(log_dir) + sizeof(_CR_BRAND_NAME) + sizeof(timestamp) + sizeof(pid) + 8];
  snprintf(logfile, sizeof(logfile), "%s/%s-%s-%d.log",
           log_dir, _CR_BRAND_NAME, timestamp, pid);

  free(state_dir);

  return logfile;
}


void 
cr_util_log_header(FILE* stream, enum cr_log_level_t lvl) {
  static const char* lvl_str[CR_LL_COUNT] = { "TRACE", "WARNING", "ERROR", "FATAL" };
  static const char* lvl_clr[CR_LL_COUNT] = {
    "\033[1;32m",   // TRACE - bright green
    "\033[1;33m",   // WARNING - yellow
    "\033[1;31m",   // ERROR - bright red
    "\033[38;5;88m" // FATAL - dark red
  };
  const char* fnt_bold  = "\033[1m";
  const char* clr_reset = "\033[0m";
  const char* clr_blue  = "\033[1;34m";

  time_t rawtime;
  struct tm timeinfo;
  // 9 = HH:MM:SS + null terminator
  char timebuf[9];  
  time(&rawtime);
  localtime_r(&rawtime, &timeinfo);
  strftime(timebuf, sizeof(timebuf), "%H:%M:%S", &timeinfo);

  bool colorize = (stream == stderr || stream == stdout); 
  fprintf(
    stream, "["_CR_BRAND_NAME"]: %s%s%s%s: %s%s%s%s: ", 
    // log level 
    colorize ? lvl_clr[lvl] : "", colorize ?  fnt_bold : "",
    lvl_str[lvl], colorize ? clr_reset : "",

    // time (blue, bold)
    colorize ? fnt_bold : "", colorize ? clr_blue : "",
    timebuf, colorize ? clr_reset : ""
  );
}



unsigned char* 
cr_util_read_file(const char* filepath, size_t* o_filesize) {
  FILE* f = fopen(filepath, "rb");
  if(!f) {
    fprintf(stderr, "corender: failed to open file: '%s'", filepath);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  size_t s = ftell(f);
  rewind(f);

  if(s <= 0) {
    fclose(f);
    return NULL;
  }

  unsigned char* data = malloc(s);
  if(!data) {
    perror("failed to allocate file data.");
    return NULL;
  }
  fread(data, 1, s, f);
  fclose(f);

  *o_filesize = s;

  return data;

}

void* 
cr_util_alloc(void* ctx, size_t n, size_t size) {
  void* ret = calloc(n, size);
  if(!ret) {
    CR_FATAL(((struct cr_context_t*)ctx)->log, "calloc() failed: Out of memory.\n");
    return NULL;
  }
  return ret;
}

const char* 
cr_util_vk_result_to_string(VkResult r)
{
  switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";

        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";

        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";

        default: return "VK_ERROR_UNKNOWN";
    }
}

uint64_t 
cr_util_get_time_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

uint32_t cr_util_djb2_hash(char *str) {
  unsigned long hash = 5381;
  int c;

  while ((c = *str++))
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;

}

bool cr_util_global_gpu_profiler_init(struct cr_context_t* ctx) {
    bool ok = cr_gpu_profiler_init(
            &global_profiler,
            ctx->phys_dev,
            ctx->logical_dev,
            ctx->graphics_queue_family, 
            CR_FRAME_COUNT,
            128,                                  // max scopes per frame
            60                                    // print every 60 collected frames
            );

    return ok;
}

struct cr_gpu_profiler_t* cr_util_global_gpu_profiler_get() {
    return &global_profiler;
}


