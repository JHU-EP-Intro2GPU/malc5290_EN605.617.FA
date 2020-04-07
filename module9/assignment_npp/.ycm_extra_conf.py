def Settings( **kwargs ):
  return {
    'flags': [ '-x', 'cuda', '-lcublas', '-std=c++11', '-Wall', '-Wextra', '-Werror' ],
  }
