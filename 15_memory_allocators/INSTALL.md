# Installation Guide for Memory Allocators

Platform-specific instructions for installing and using different memory allocators.

## Ubuntu/Debian Linux

### Install from Package Manager

```bash
# tcmalloc (Google Performance Tools)
sudo apt-get update
sudo apt-get install libgoogle-perftools-dev libtcmalloc-minimal4

# jemalloc
sudo apt-get install libjemalloc-dev libjemalloc2

# Verify installations
dpkg -L libtcmalloc-minimal4 | grep "\.so"
dpkg -L libjemalloc2 | grep "\.so"
```

### Install mimalloc from Source

```bash
# Install dependencies
sudo apt-get install cmake build-essential

# Clone and build
git clone https://github.com/microsoft/mimalloc.git
cd mimalloc
mkdir -p build
cd build
cmake ..
make -j$(nproc)
sudo make install

# Library will be at /usr/local/lib/libmimalloc.so
sudo ldconfig
```

## Red Hat/CentOS/Fedora

```bash
# tcmalloc
sudo dnf install gperftools-devel gperftools-libs

# jemalloc
sudo dnf install jemalloc-devel jemalloc

# mimalloc (build from source as above)
```

## macOS

### Using Homebrew

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# tcmalloc
brew install gperftools

# jemalloc
brew install jemalloc

# mimalloc
brew install mimalloc

# Verify installations
ls -l /usr/local/lib/libtcmalloc.*
ls -l /usr/local/lib/libjemalloc.*
ls -l /usr/local/lib/libmimalloc.*
```

## Windows

### Using vcpkg

```powershell
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install allocators
.\vcpkg install mimalloc:x64-windows
.\vcpkg install jemalloc:x64-windows

# Integrate with Visual Studio
.\vcpkg integrate install
```

### Manual Build (mimalloc)

```powershell
# Clone repository
git clone https://github.com/microsoft/mimalloc.git
cd mimalloc

# Build with Visual Studio
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# DLL will be in build/Release/
```

## Using Allocators

### Method 1: Link at Compile Time

```bash
# With tcmalloc
g++ -o myapp myapp.cpp -ltcmalloc

# With jemalloc  
g++ -o myapp myapp.cpp -ljemalloc

# With mimalloc
g++ -o myapp myapp.cpp -lmimalloc

# Then run normally
./myapp
```

### Method 2: LD_PRELOAD (Linux)

```bash
# No recompilation needed!

# With tcmalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./myapp

# With jemalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./myapp

# With mimalloc
LD_PRELOAD=/usr/local/lib/libmimalloc.so ./myapp

# Make it permanent in script
cat > run_with_tcmalloc.sh << 'SCRIPT'
#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
exec "$@"
SCRIPT
chmod +x run_with_tcmalloc.sh
./run_with_tcmalloc.sh ./myapp
```

### Method 3: DYLD_INSERT_LIBRARIES (macOS)

```bash
# With tcmalloc
DYLD_INSERT_LIBRARIES=/usr/local/lib/libtcmalloc.dylib ./myapp

# With jemalloc
DYLD_INSERT_LIBRARIES=/usr/local/lib/libjemalloc.dylib ./myapp

# With mimalloc
DYLD_INSERT_LIBRARIES=/usr/local/lib/libmimalloc.dylib ./myapp

# Note: May require disabling SIP for some apps
```

### Method 4: Windows DLL

```cpp
// myapp.cpp - Link mimalloc statically
#include <mimalloc.h>

// Or load DLL dynamically
#pragma comment(lib, "mimalloc.lib")

int main() {
    // Automatically uses mimalloc for malloc/free
    void* ptr = malloc(1024);
    free(ptr);
    return 0;
}
```

## Verification

### Check Which Allocator is Being Used

```bash
# Linux - Check loaded libraries
ldd ./myapp | grep -E 'tcmalloc|jemalloc|mimalloc'

# Linux - Runtime check with ltrace
ltrace -e malloc,free ./myapp 2>&1 | head -20

# macOS - Check loaded libraries
otool -L ./myapp | grep -E 'tcmalloc|jemalloc|mimalloc'

# Show memory maps
cat /proc/$(pgrep myapp)/maps | grep -E 'tcmalloc|jemalloc|mimalloc'
```

### Verify Performance

```cpp
// test_allocator.cpp
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    using namespace std::chrono;
    
    const int N = 1000000;
    auto start = high_resolution_clock::now();
    
    std::vector<void*> ptrs;
    for (int i = 0; i < N; i++) {
        ptrs.push_back(malloc(64));
    }
    for (void* p : ptrs) {
        free(p);
    }
    
    auto end = high_resolution_clock::now();
    auto ms = duration_cast<milliseconds>(end - start).count();
    
    std::cout << "Time: " << ms << " ms\n";
    std::cout << "Ops/sec: " << (N * 1000.0 / ms) << "\n";
    
    return 0;
}
```

```bash
# Build and test
g++ -O3 -o test_allocator test_allocator.cpp

# Test default
./test_allocator

# Test tcmalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./test_allocator

# Test jemalloc  
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./test_allocator
```

## Docker Setup

```dockerfile
# Dockerfile with all allocators
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgoogle-perftools-dev \
    libtcmalloc-minimal4 \
    libjemalloc-dev \
    libjemalloc2

# Build mimalloc
RUN git clone https://github.com/microsoft/mimalloc.git && \
    cd mimalloc && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && \
    ldconfig

# Set working directory
WORKDIR /app

# Default to bash
CMD ["/bin/bash"]
```

```bash
# Build and run
docker build -t allocator-test .
docker run -it -v $(pwd):/app allocator-test

# Inside container, test allocators
g++ -O3 -o test test.cpp
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ./test
```

## Troubleshooting

### Library Not Found

```bash
# Find the library
find /usr -name "libtcmalloc.so*" 2>/dev/null
find /usr -name "libjemalloc.so*" 2>/dev/null

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Or run ldconfig
sudo ldconfig
```

### Wrong Version Loaded

```bash
# Check library dependencies
ldd /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# Force specific version
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9 ./myapp
```

### Conflicts with Other Libraries

```bash
# Some libraries bundle their own allocator
# Check for conflicts
nm -D ./myapp | grep malloc

# May need to link statically
g++ -o myapp myapp.cpp -Wl,-Bstatic -ltcmalloc -Wl,-Bdynamic
```

### macOS SIP Issues

```bash
# System Integrity Protection may block DYLD_INSERT_LIBRARIES

# Option 1: Disable SIP (not recommended)
csrutil disable  # In recovery mode

# Option 2: Link at compile time instead
g++ -o myapp myapp.cpp -L/usr/local/lib -ltcmalloc

# Option 3: Build app without hardened runtime
```

## Production Deployment

### Systemd Service (Linux)

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application
After=network.target

[Service]
Type=simple
User=myapp
Environment="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
ExecStart=/usr/local/bin/myapp
Restart=always

[Install]
WantedBy=multi-user.target
```

### Container Deployment

```dockerfile
# Production Dockerfile
FROM ubuntu:22.04 AS build

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjemalloc-dev

# Copy source
COPY . /app
WORKDIR /app

# Build with jemalloc
RUN g++ -O3 -o myapp main.cpp -ljemalloc

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libjemalloc2
COPY --from=build /app/myapp /usr/local/bin/
CMD ["/usr/local/bin/myapp"]
```

### Configuration Management

```bash
# Ansible playbook example
- name: Install jemalloc
  apt:
    name: libjemalloc2
    state: present

- name: Configure application to use jemalloc
  lineinfile:
    path: /etc/environment
    line: 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2'
```

## Summary

**Quick Start:**
1. Ubuntu: `sudo apt-get install libjemalloc2`
2. macOS: `brew install jemalloc`
3. Use: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./myapp`

**Best Practices:**
- Test with LD_PRELOAD before recompiling
- Benchmark your specific workload
- Monitor memory usage in production
- Have rollback plan
