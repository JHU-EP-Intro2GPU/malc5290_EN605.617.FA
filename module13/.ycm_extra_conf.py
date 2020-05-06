# clang++ -I/usr/local/cuda/include/ HelloWorld.cpp -lOpenCL
def Settings( **kwargs ):
  return {
    'flags': [ '-x', '-I/usr/local/cuda/include/', '-lOpenCL', '-Wall', '-Wextra', '-Werror' ],
  }
