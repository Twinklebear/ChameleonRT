# ChameleonRT

An example path tracer which runs on multiple ray tracing backends (Embree/DXR/OptiX/Vulkan).
Uses [tinyobjloader](https://github.com/syoyo/tinyobjloader) to load OBJ files. The San Miguel,
Sponza and Rungholt models shown below are from Morgan McGuire's [Computer Graphics Data Archive](https://casual-effects.com/data/).

[![San Miguel, Sponza and Rungholt](https://i.imgur.com/tKZYjzn.jpg)](https://i.imgur.com/pVhQK3j.jpg)

## Ray Tracing Backends  

The currently implemented backends are: Embree, DXR, OptiX, and Vulkan.
When running the program, you can pick which backend you want from
those you compiled with on the command line. Running the program with no
arguments will print help information.

```
./chameleonrt <backend> <mesh.obj>
```

All three ray tracing backends use [SDL2](https://www.libsdl.org/index.php) for window management
and [GLM](https://glm.g-truc.net/0.9.9/index.html) for math.
If CMake doesn't find your SDL2 install you can point it to the root
of your SDL2 directory by passing `-DSDL2=<path>`.
Similarly for GLM, you can point it to the glmConfig.cmake file
in your GLM distribution by passing `-Dglm_DIR=<path>`.
To track and report statistics about the number of rays traced per-second
run CMake with `-DREPORT_RAY_STATS=ON`. Tracking these statistics can
impact performance slightly.

ChameleonRT only supports per-OBJ group/mesh materials, OBJ files using per-face materials
can be reexported from Blender with the "Material Groups" option enabled.

### Embree

Dependencies: [Embree](https://embree.github.io/),
[TBB](https://www.threadingbuildingblocks.org/) and [ISPC](https://ispc.github.io/).

To build the Embree backend run CMake with:

```
cmake .. -DENABLE_EMBREE=ON \
	-Dembree_DIR=<path to embree-config.cmake> \
	-DTBB_DIR=<path TBBConfig.cmake> \
	-DISPC_DIR=<path to ispc>
```

You can then pass `-embree` to use the Embree backend. The `TBBConfig.cmake` will
be under `<tbb root>/cmake`, while `embree-config.cmake` is in the root of the
Embree directory.

### OptiX

Dependencies: [OptiX 7](https://developer.nvidia.com/optix), [CUDA 10](https://developer.nvidia.com/cuda-zone).

To build the OptiX backend run CMake with:

```
cmake .. -DENABLE_OPTIX=ON
```

You can then pass `-optix` to use the OptiX backend.

If CMake doesn't find your install of OptiX you can tell it where
it's installed with `-DOptiX_INSTALL_DIR`.

### DirectX Ray Tracing

If you're on Windows 10 1809, have the 10.0.17763 SDK and a DXR capable GPU you can also run
the DirectX Ray Tracing backend.

To build the DXR backend run CMake with:

```
cmake .. -DENABLE_DXR=ON
```

You can then pass `-dxr` to use the DXR backend.

### Vulkan NV Ray Tracing

Dependencies: [Vulkan](https://vulkan.lunarg.com/).

To build the Vulkan backend run CMake with:

```
cmake .. -DENABLE_VULKAN=ON
```

You can then pass `-vulkan` to use the Vulkan backend.

If CMake doesn't find your install of Vulkan you can tell it where it's
installed with `-DVULKAN_SDK`. This path should be to the specific version
of Vulkan, for example: `-DVULKAN_SDK=<path>/VulkanSDK/1.1.114.0/`

### OSPRay [stale]

The OSPRay reference backend has gotten stale during the path tracer implementation
in the other backends, but is on the list to get back up to date.

Dependencies: [OSPRay](http://www.ospray.org/).

To build the OSPRay backend run CMake with:

```
cmake .. -DENABLE_OSPRAY=ON -Dospray_DIR=<path to osprayConfig.cmake>
```

You can then pass `-ospray` to use the OSPRay backend.

## Citation

If you find ChameleonRT useful in your work, please cite it as:

```
@misc{chameleonrt,
	author = {Will Usher},
	year = {2019},
	note = {https://github.com/Twinklebear/ChameleonRT},
	title = {{ChameleonRT}}
} 
```
