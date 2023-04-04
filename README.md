# ChameleonRT

[![CMake](https://github.com/Twinklebear/ChameleonRT/actions/workflows/cmake.yml/badge.svg)](https://github.com/Twinklebear/ChameleonRT/actions/workflows/cmake.yml)

An example path tracer that runs on multiple ray tracing backends (Embree/Embree4+SYCL/DXR/OptiX/Vulkan/Metal/OSPRay).
Uses [tinyobjloader](https://github.com/syoyo/tinyobjloader) to load OBJ files,
[tinygltf](https://github.com/syoyo/tinygltf) to load glTF files and, optionally,
Ingo Wald's [pbrt-parser](https://github.com/ingowald/pbrt-parser) to load PBRTv3 files.
The San Miguel,
Sponza and Rungholt models shown below are from Morgan McGuire's [Computer Graphics Data Archive](https://casual-effects.com/data/).

Binaries built for each platform with support for all rendering backends available on that platform
can be downloaded from the [Releases](https://github.com/Twinklebear/ChameleonRT/releases) or the latest [Actions](https://github.com/Twinklebear/ChameleonRT/actions) artifacts.
On Linux set the `LD_LIBRARY_PATH` to include the directory where you extract the
application, on macOS do the same for `DYLD_LIBRARY_PATH`. On macOS you may need to do some
tweaking/security allowances to get the application to run. Only x86_64 binaries are provided for macOS at the moment.

[![San Miguel, Sponza and Rungholt](https://i.imgur.com/tKZYjzn.jpg)](https://i.imgur.com/pVhQK3j.jpg)

## Controls

The camera is an arcball camera that moves around the camera's focal point.

- Zoom in/out using the mousewheel. (Two-finger scroll motion on MacBook Pro etc. trackpad.)
- Click and drag to rotate.
- Right-click and drag to pan. (Two-finger click and drag on MacBook Pro etc. trackpad.)

Keys while the application window is in focus:

- Print the current camera position, center point, up vector and field of view (FOV) to the terminal by pressing the `p` key.
- Save image by pressing the `s` key.

## Command-line Options

```text
-eye <x> <y> <z>       Set the camera position
-center <x> <y> <z>    Set the camera focus point
-up <x> <y> <z>        Set the camera up vector
-fov <fovy>            Specify the camera field of view (in degrees)
-camera <n>            If the scene contains multiple cameras, specify which
                       should be used. Defaults to the first camera
-img <x> <y>           Specify the window dimensions. Defaults to 1280x720
```

## Ray Tracing Backends  

The currently implemented backends are: Embree, DXR, OptiX, Vulkan, and Metal.
When running the program, you can pick which backend you want from
those you compiled with on the command line. Running the program with no
arguments will print help information.

```
./chameleonrt <backend> <mesh.obj/gltf/glb>
```

All five ray tracing backends use [SDL2](https://www.libsdl.org/index.php) for window management
and [GLM](https://github.com/g-truc/glm) for math.
If CMake doesn't find your SDL2 install you can point it to the root
of your SDL2 directory by passing `-DSDL2_DIR=<path>`. GLM will be automatically
downloaded by CMake during the build process.

To track statistics about the number of rays traced per-second
run CMake with `-DREPORT_RAY_STATS=ON`. Tracking these statistics can
impact performance slightly.

ChameleonRT only supports per-OBJ group/mesh materials, OBJ files using per-face materials
can be reexported from Blender with the "Material Groups" option enabled.

To build with PBRT file support set the CMake option `CHAMELEONRT_PBRT_SUPPORT=ON` and pass
`-DpbrtParser_DIR=<path>` with `<path>` pointing to the CMake export files for
your build of [Ingo Wald's pbrt-parser](https://github.com/ingowald/pbrt-parser).

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

You can then pass `embree` to use the Embree backend. The `TBBConfig.cmake` will
be under `<tbb root>/cmake`, while `embree-config.cmake` is in the root of the
Embree directory.

### Embree4 + SYCL

Dependencies: [Embree 4](https://embree.github.io/),
[TBB](https://www.threadingbuildingblocks.org/) and Intel's oneAPI SYCL compiler.
Currently (4/4/2023) the Embree4 + SYCL backend requires the
[20230304 nightly](https://github.com/intel/llvm/releases/tag/sycl-nightly%2F20230304) build of
the oneAPI SYCL compiler and the latest Intel Arc GPU drivers.
I have tested with driver version 31.0.101.4255 on Windows, it seems that the Ubuntu
drivers are on an older version with their last release showing as being in Oct 2022.

To build the Embree4 + SYCL backend run CMake with:

```
cmake .. -G Ninja `
    -DCMAKE_C_COMPILER=<path to dpcpp nightly>/bin/clang.exe `
    -DCMAKE_CXX_COMPILER=<path to dpcpp nightly>/bin/clang++.exe `
    -DENABLE_EMBREE_SYCL=ON `
    -Dembree_DIR=<path to embree-config.cmake> `
	-DTBB_DIR=<path TBBConfig.cmake>
cmake --build . --config relwithdebinfo
```

You can then pass `embree_sycl` to use the Embree4 + SYCL backend. The `TBBConfig.cmake` will
be under `<tbb root>/cmake`, while `embree-config.cmake` is in the root of the
Embree directory.

Note that building the Embree4 + SYCL backend is currently incompatible with OptiX
since the compiler is not supported by CUDA. You can simply build the OptiX backend
(or the Embree4 + SYCL) backend in a separate build directory to get builds of both.

### OptiX

Dependencies: [OptiX 7.2](https://developer.nvidia.com/optix), [CUDA 11](https://developer.nvidia.com/cuda-zone).

To build the OptiX backend run CMake with:

```
cmake .. -DENABLE_OPTIX=ON
```

You can then pass `optix` to use the OptiX backend.

If CMake doesn't find your install of OptiX you can tell it where
it's installed with `-DOptiX_INSTALL_DIR`.

### DirectX Ray Tracing

If you're on Windows 10 1809 or higher, have the latest Windows 10 SDK installed and a DXR
capable GPU you can also run the DirectX Ray Tracing backend.

To build the DXR backend run CMake with:

```
cmake .. -DENABLE_DXR=ON
```

You can then pass `dxr` to use the DXR backend.

### Vulkan KHR Ray Tracing

Dependencies: [Vulkan](https://vulkan.lunarg.com/) (SDK 1.2.162 or higher)

To build the Vulkan backend run CMake with:

```
cmake .. -DENABLE_VULKAN=ON
```

You can then pass `vulkan` to use the Vulkan backend.

If CMake doesn't find your install of Vulkan you can tell it where it's
installed with `-DVULKAN_SDK`. This path should be to the specific version
of Vulkan, for example: `-DVULKAN_SDK=<path>/VulkanSDK/<version>/`

### Metal

Dependencies: [Metal](https://developer.apple.com/metal/) and a macOS 11+ device that supports ray tracing.

To build the Metal backend run CMake with:

```
cmake .. -DENABLE_METAL=ON
```

You can then pass `metal` to use the Metal backend.

### OSPRay

Dependencies: [OSPRay 2.0](http://www.ospray.org/), [TBB](https://www.threadingbuildingblocks.org/).

To build the OSPRay backend run CMake with:

```
cmake .. -DENABLE_OSPRAY=ON -Dospray_DIR=<path to osprayConfig.cmake>
```

You may also need to specify OSPRay's dependencies,
[ospcommon](https://github.com/ospray/ospcommon) and [OpenVKL](https://github.com/openvkl/openvkl),
depending on how you got or built the OSPRay binaries.
If you downloaded the OSPRay release binaries, you just need to
specify that path.

You can then pass `ospray` to use the OSPRay backend.

## Citation

If you find ChameleonRT useful in your work, please cite it as:

```bibtex
@misc{chameleonrt,
	author = {Will Usher},
	year = {2019},
	howpublished = {\url{https://github.com/Twinklebear/ChameleonRT}},
	title = {{ChameleonRT}}
} 
```
