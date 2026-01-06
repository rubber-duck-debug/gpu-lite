# Build Scripts

This directory contains platform-specific build scripts for gpulite examples.

## Scripts

### `build_examples_linux.sh`
- **Platform**: Linux
- **Requirements**: GCC 7+, Clang 5+, CMake 3.12+
- **Usage**: `./scripts/build_examples_linux.sh` (from project root)
- **Description**: Builds all gpulite examples using CMake and make

### `build_examples_windows.bat`
- **Platform**: Windows
- **Requirements**: Visual Studio 2017+, CMake 3.12+
- **Usage**: `scripts\build_examples_windows.bat` (from project root)
- **Description**: Builds all gpulite examples using CMake and Visual Studio

## Usage

All scripts should be run from the gpulite root directory:

```bash
# Linux
./scripts/build_examples_linux.sh

# Windows (Command Prompt)
scripts\build_examples_windows.bat

# Windows (PowerShell)
.\scripts\build_examples_windows.bat
```

## Output

Both scripts will:
1. Create build directories for each example
2. Configure CMake for each example project
3. Build the executables
4. Report the location of the built executables

The executables will be located in:
- `examples/basic_vector_add/build/`
- `examples/matrix_multiply/build/`
- `examples/templated_kernels/build/`