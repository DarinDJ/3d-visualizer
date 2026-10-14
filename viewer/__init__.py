"""A small from-scratch 3-D model viewer.

Modules
    camera     - camera module: view / projection matrices and an orbit camera
    loader     - object loader: OBJ, STL and PLY readers + writers
    mesh       - indexed triangle mesh container and normal computation
    controls   - mouse manipulation: turns mouse events into camera operations
    renderer   - OpenGL 3.3 rendering (shaders, buffers, grid, axes)
    app        - GLFW window, event loop and keyboard shortcuts
"""

__version__ = "1.0.0"
