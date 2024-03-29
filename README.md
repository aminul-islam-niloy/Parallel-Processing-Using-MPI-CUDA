﻿# Parallel-Processing-Using-MPI-CUDA
Certainly! Here's a simple template for a README file for your Parallel Processing Lab:

# Parallel Processing Lab

## Table of Contents

1. [Matrix Multiplication](#matrix-multiplication)
2. [Word Count and Sorting](#word-count-and-sorting)
3. [Phonebook Search](#phonebook-search)
4. [Pattern Occurrence Count](#pattern-occurrence-count)

## Matrix Multiplication

### Using MPI

```bash
mpiexec -n <number_of_processes> ./matrix_multiplication_mpi <K> <M> <N> <P>
```

- `<number_of_processes>`: Number of MPI processes.
- `<K>`, `<M>`, `<N>`, `<P>`: Matrix dimensions.

### Using CUDA

```bash
./matrix_multiplication_cuda <K> <M> <N> <P>
```

- `<K>`, `<M>`, `<N>`, `<P>`: Matrix dimensions.

Output: Time taken for multiplication.

## Word Count and Sorting

### Using MPI

```bash
mpiexec -n <number_of_processes> ./word_count_mpi <filename>
```

- `<number_of_processes>`: Number of MPI processes.
- `<filename>`: Input file containing text.

### Using CUDA

```bash
./word_count_cuda <filename>
```

- `<filename>`: Input file containing text.

Output: Total time, top 10 occurrences.

## Phonebook Search

### Using MPI

```bash
mpiexec -n <number_of_processes> ./phonebook_search_mpi <filename> <search_name>
```

- `<number_of_processes>`: Number of MPI processes.
- `<filename>`: Phonebook file.
- `<search_name>`: Name to search for.

Output: Total time, matching names and contact numbers.

## Pattern Occurrence Count

### Using MPI

```bash
mpiexec -n <number_of_processes> ./pattern_occurrence_mpi <filename> <pattern>
```

- `<number_of_processes>`: Number of MPI processes.
- `<filename>`: Input file containing text.
- `<pattern>`: Pattern to search for.

### Using CUDA

```bash
./pattern_occurrence_cuda <filename> <pattern>
```

- `<filename>`: Input file containing text.
- `<pattern>`: Pattern to search for.
