# CO - Unified Collective Communication (CO-UCC)

### CO-UCC: Performance-Enhanced Unified Collective Communication Library

### Overview

CO-UCC is a high-performance extension of NVIDIA's Unified Collective Communication (UCC) library, designed to enhance communication throughput for collective operations across heterogeneous systems. Inspired by the COCCL library, CO-UCC integrates precision and compression techniques to accelerate communication on diverse hardware platforms, including CPUs, GPUs, and DPUs.
This project builds upon the UCC framework introduced in the paper by Venkata et al. (2024) and draws inspiration from the COCCL library (HPDPS Group, 2025). CO-UCC aims to address performance limitations in the original UCC implementation while maintaining its platform-agnostic flexibility.

### Features

Performance Optimization: Incorporates advanced compression and precision techniques to improve communication efficiency.
Platform Agnostic: Supports diverse hardware platforms, including CPUs, GPUs, and DPUs, leveraging UCC's flexible framework.
Collective Operations: Enhances throughput for various collective communication operations critical for high-performance computing (HPC).
Compatibility: Built as an extension of the UCC library, ensuring seamless integration with existing UCC-based applications.

### TODO: 
1.realize the AlltoAll and Allreduce collective communication

2.realize the UCX compress support  

### Installation

To be added: Installation instructions for CO-UCC, including dependencies and build steps.

### Usage

To be added: Example code snippets and usage instructions for integrating CO-UCC into your HPC applications.

### References

Venkata, M. G., Petrov, V., Lebedev, S., Bureddy, D., Aderholdt, F., Ladd, J., Bloch, G., Dubman, M., & Shainer, G. (2024). Unified Collective Communication (UCC): An Unified Library for CPU, GPU, and DPU Collectives. IEEE Symposium on High-Performance Interconnects (HOTI 2024). https://doi.org/10.1109/HOTI63208.2024.00018

HPDPS Group. (2025). COCCL: Compression and Precision Co-Aware Collective Communication Library. https://github.com/hpdps-group/coccl

Shamis, P., Venkata, M. G., Lopez, M. G., Baker, M. B., Hernandez, O., Itigin, Y., Dubman, M., Shainer, G., Graham, R. L., Liss, L., & others. (2015). UCX: An Open Source Framework for HPC Network APIs and Beyond. IEEE 23rd Annual Symposium on High-Performance Interconnects.


### License
To be added: License information for the CO-UCC library.

