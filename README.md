# kernel_in_kernel
Benchmarking Idea for AGIHouse HardCore Hackathon CUDA edition

We're going to model a thing the Linux kernel does where it has an array of structs and it iterates that array to extract a value from each struct reducing to a new u64 array of the same length.

Looking at struct page/folio struct page can be 96 bytes
