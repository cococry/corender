project(
  'corender',
  ['c', 'cpp'],
  version : '0.1.0',
  default_options : [
    'warning_level=2',
    'c_std=c11',
    'cpp_std=c++17'
  ]
)

cc  = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

if get_option('buildtype') == 'debug'
  common_cflags = ['-g', '-O0', '-DDEBUG']
elif get_option('buildtype') == 'release'
  common_cflags = [
    '-O3',
    '-DNDEBUG',
    '-march=native',
    '-flto',
    '-fno-plt',
    '-fomit-frame-pointer',
    '-fstrict-aliasing'
  ]
else
  common_cflags = []
endif

add_project_arguments(common_cflags, language : ['c', 'cpp'])
add_project_link_arguments(common_cflags, language : ['c', 'cpp'])

corender_sources = files(
  'src/corender.c',
  'src/util.c',
  'src/pipeline.c',
  'src/mem.c'
)

vma_impl = static_library(
  'vma_impl',
  'vendor/vma/vma_impl.cpp'
)

corender_lib = static_library(
  'corender',
  corender_sources,
  link_with : vma_impl,
  install : true
)

install_headers(
  run_command('sh', '-c', 'ls include/corender/*.h', check : true)
    .stdout()
    .strip()
    .split(),
  subdir : 'corender'
)

glfw_dep = dependency('glfw3', required : false)
gl_dep   = cc.find_library('GL', required : false)
vk_dep   = dependency('vulkan', required : false)

example_sources = run_command(
  'sh', '-c', 'ls examples/*.c', check : true
).stdout().strip().split()

foreach src : example_sources
  name = src.split('/')[-1].split('.')[0]

  executable(
    name,
    src,
    link_with : corender_lib,
    dependencies : [glfw_dep, gl_dep, vk_dep],
    link_args : ['-lstdc++'],
    install : false
  )
endforeach

fs = import('fs')
glslc = find_program('glslc')

shader_root = 'shaders'

foreach dir : fs.listdir(shader_root)
  shader_dir = shader_root / dir

  if not fs.is_dir(shader_dir)
    continue
  endif


  foreach shader : fs.listdir(shader_dir)
    if not shader.endswith('.glsl')
      continue
    endif

    stage = ''
    if shader.contains('vert')
      stage = 'vertex'
    elif shader.contains('frag')
      stage = 'fragment'
    else
      error('Unknown shader stage: ' + shader)
    endif

    custom_target(
      dir + '_' + shader,
      input  : shader_dir / shader,
      output : shader.replace('.glsl', '.spv'),
      command : [
        glslc,
        '-fshader-stage=' + stage,
        '@INPUT@',
        '-o',
        '@OUTPUT@',
        ],
      build_by_default : true
      )
  endforeach
endforeach
