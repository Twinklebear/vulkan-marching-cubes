# Vulkan Marching Cubes

This is GPU-parallel implementation of Marching Cubes using Vulkan. It builds
using CMake, and you can run it via:
```
./vulkan_marching_cubes <volume_file.raw> <isovalue> [optional output.obj]
```

The program takes volumes whose file names are formatted like those found
on [OpenScivisDatasets](https://klacansky.com/open-scivis-datasets/), you can
download datasets from that page to try out the app. For example, you can
compute the isosurface at isovalue = 80 on the skull and output the mesh
to an OBJ file:
```
./vulkan_marching_cubes skull_256x256x256_uint8.raw 80 skull_iso_80.obj
```
