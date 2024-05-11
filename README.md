

---

# Sparse Matrix-Dense Matrix Multiplication

This project demonstrates the multiplication of a sparse matrix with a dense matrix using various methods on the CPU and GPU. The methods include computation on the CPU, a single-threaded approach on the GPU, a multi-threaded approach on the GPU, and an optimized version using shared memory on the GPU.

## Project Structure

- `functions.h`: Header file that declares all the matrix multiplication functions.
- `cpu.c`: Contains the implementation of the matrix multiplication on the CPU.
- `single_thread.cu`: Contains the CUDA implementation for a single-threaded GPU matrix multiplication.
- `multi_thread.cu`: Contains the CUDA implementation for a multi-threaded GPU matrix multiplication using global memory.
- `shared_memory.cu`: Contains the CUDA implementation for a multi-threaded GPU matrix multiplication using shared memory.
- `main.cu`: The main entry point of the program that calls the multiplication functions and measures their performance.

## Building the Project

This project uses `nvcc`, the NVIDIA CUDA compiler driver, for building the CUDA programs. A `Makefile` is provided for convenience to build the project.

### Prerequisites

Ensure you have the following installed:
- NVIDIA CUDA Toolkit (check with `nvcc --version`)
- A compatible C compiler (GCC for Linux systems)

### Compile Instructions

To build all the executable files, use the following command in the project's root directory:

```bash
make all
```

This command compiles all the source files and links them into a single executable named `main`.

## Running the Program

After building the project, you can run the program by executing:

```bash
./main
```

This will run the matrix multiplication using the different methods and print out the execution times for each.

## Expected Output

The program will display the computation time for each method:
- CPU computation time
- Single-thread GPU computation time
- Multi-thread GPU computation time
- Shared memory GPU computation time

## Cleaning the Build

To clean up the build files and executables, you can use:

```bash
make clean
```

This command will remove all the compiled object files and the executable.

---
