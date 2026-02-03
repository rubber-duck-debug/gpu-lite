@echo off
REM Windows build script for gpulite examples
REM Requires Visual Studio with C++17 support
REM Run this script from the gpulite root directory

echo Building gpulite examples for Windows...

REM Change to root directory (assuming script is in scripts/ folder)
cd /d "%~dp0.."

REM Create build directories if they don't exist
if not exist "examples\basic_vector_add\build" mkdir "examples\basic_vector_add\build"
if not exist "examples\matrix_multiply\build" mkdir "examples\matrix_multiply\build"
if not exist "examples\templated_kernels\build" mkdir "examples\templated_kernels\build"

REM Build basic_vector_add
echo Building basic_vector_add...
cd examples\basic_vector_add\build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
cd ..\..\..

REM Build matrix_multiply
echo Building matrix_multiply...
cd examples\matrix_multiply\build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
cd ..\..\..

REM Build templated_kernels
echo Building templated_kernels...
cd examples\templated_kernels\build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
cd ..\..\..

echo Build complete!
echo.
echo Executables are located in:
echo - examples\basic_vector_add\build\Release\vector_add.exe
echo - examples\matrix_multiply\build\Release\matrix_multiply.exe
echo - examples\templated_kernels\build\Release\templated_kernels.exe
