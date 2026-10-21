#!/usr/bin/env bash
# Build a portable libedgetpu in a container: out/direct/k8 (max) and out/throttled/k8 (std).
#
# abseil and flatbuffers are linked in statically, so the only runtime
# dependencies are glibc >= 2.36, libstdc++ and libusb-1.0.
#
# Usage: makefile_build/build-static.sh        (ENGINE=docker to use docker)
set -euo pipefail

ENGINE=${ENGINE:-podman}
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=libedgetpu-static-build

"$ENGINE" build -t "$IMAGE" -f "$REPO/makefile_build/Containerfile" "$REPO/makefile_build"

# Static archives must follow the objects; the group lets them resolve each other.
LDLIBS='-Wl,--start-group /usr/local/lib/libflatbuffers.a /usr/local/lib/libabsl_*.a -Wl,--end-group -lusb-1.0'

"$ENGINE" run --rm -v "$REPO":/src:Z -w /src/makefile_build "$IMAGE" bash -c "
    make clean
    make -j\$(nproc) TFROOT=/tensorflow LIBEDGETPU_LDLIBS='$LDLIBS' libedgetpu libedgetpu-throttled
"

# A shared library may link with symbols left undefined; make sure none are abseil/flatbuffers.
for lib in "$REPO"/out/{direct,throttled}/k8/libedgetpu.so.1.0; do
    if nm -D --undefined-only "$lib" | grep -E -q 'absl|flatbuffers'; then
        echo "error: $lib still needs abseil/flatbuffers at runtime" >&2
        exit 1
    fi
done
echo "Built:"
ls -l "$REPO"/out/{direct,throttled}/k8/libedgetpu.so.1.0
