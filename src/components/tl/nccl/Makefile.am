# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

if TL_NCCL_ENABLED

###############################
# 子模块：Allgatherv 算子源码
###############################
allgatherv =                      \
    allgatherv/allgatherv.h       \
    allgatherv/allgatherv.c

###############################
# 子模块：压缩相关源码（你的代码）
###############################
compress =                        \
    compress/base_tool.h          \
    compress/cuda_checks.h        \
    compress/get_minmax.h         \
    compress/get_minmax.cu        \
    compress/compresskernel.h     \
    compress/compresskernel.cu    \
    compress/compress_nccl.h      \
    compress/compress_nccl.cc

###############################
# NCCL TL 核心源码
###############################
sources =                         \
    tl_nccl.h                     \
    tl_nccl.c                     \
    tl_nccl_lib.c                 \
    tl_nccl_context.c             \
    tl_nccl_team.c                \
    tl_nccl_coll.h                \
    tl_nccl_coll.c                \
    $(allgatherv)                 \
    $(compress)

###############################
# 构建动态库规则
###############################
module_LTLIBRARIES = libucc_tl_nccl.la

# 编译源文件
libucc_tl_nccl_la_SOURCES  = $(sources)

# 预处理器标志（包含 CUDA/NCCL）
libucc_tl_nccl_la_CPPFLAGS = \
    $(AM_CPPFLAGS)           \
    $(BASE_CPPFLAGS)         \
    $(CUDA_CPPFLAGS)         \
    $(NCCL_CPPFLAGS)

# C 语言编译标志
libucc_tl_nccl_la_CFLAGS   = $(BASE_CFLAGS)

# 链接参数（版本号 + CUDA + NCCL）
libucc_tl_nccl_la_LDFLAGS  = \
    -version-info $(SOVERSION) --as-needed \
    $(CUDA_LDFLAGS) $(NCCL_LDFLAGS)

# 链接库依赖（CUDA/NCCL/UCC 主库）
libucc_tl_nccl_la_LIBADD   = \
    $(CUDA_LIBS) $(NCCL_LIBADD) \
    $(UCC_TOP_BUILDDIR)/src/libucc.la

# 引入统一模块规则
include $(top_srcdir)/config/module.am

###############################
# 手动指定 .cu 文件的编译规则
###############################

# CUDA 编译器
NVCC = nvcc

# CUDA 编译选项（确保位置无误）
NVCCFLAGS = -O2 -Xcompiler -fPIC -std=c++11

# .cu 文件需要手动编译为 .lo 对象文件
compress/get_minmax.lo: compress/get_minmax.cu
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

compress/compresskernel.lo: compress/compresskernel.cu
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

endif
