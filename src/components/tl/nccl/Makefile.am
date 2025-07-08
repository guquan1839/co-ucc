if TL_NCCL_ENABLED

###############################
# 子模块：Allgatherv 算子源码
###############################
allgatherv =                      \
    allgatherv/allgatherv.h       \
    allgatherv/allgatherv.c

###############################
# 子模块：压缩模块（含 CUDA/C++）
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
# 构建动态库
###############################
module_LTLIBRARIES = libucc_tl_nccl.la

libucc_tl_nccl_la_SOURCES  = $(sources)

libucc_tl_nccl_la_CPPFLAGS = \
    $(AM_CPPFLAGS)           \
    $(BASE_CPPFLAGS)         \
    $(CUDA_CPPFLAGS)         \
    $(NCCL_CPPFLAGS)

libucc_tl_nccl_la_CFLAGS   = $(BASE_CFLAGS)

libucc_tl_nccl_la_LDFLAGS  = \
    -version-info $(SOVERSION) --as-needed \
    $(CUDA_LDFLAGS) $(NCCL_LDFLAGS)

libucc_tl_nccl_la_LIBADD   = \
    $(CUDA_LIBS) $(NCCL_LIBADD) \
    $(UCC_TOP_BUILDDIR)/src/libucc.la


###############################
# CUDA 自动编译规则（使用 cuda_lt.sh）
###############################
NVCC = nvcc

NVCCFLAGS = -O2 -std=c++11 \
    -Xcompiler=-Wall \
    -Xcompiler=-Werror \
    -Xcompiler=-funwind-tables \
    -Xcompiler=-Wno-missing-field-initializers \
    -Xcompiler=-Wno-unused-parameter \
    -Xcompiler=-Wno-long-long \
    -Xcompiler=-Wno-endif-labels \
    -Xcompiler=-Wno-sign-compare \
    -Xcompiler=-Wno-multichar \
    -Xcompiler=-Wno-deprecated-declarations \
    -Xcompiler=-Winvalid-pch \
    -Xcompiler=-Wshadow

.cu.o:
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

.cu.lo:
	/bin/bash $(top_srcdir)/nccl_cuda_lt.sh "$(LIBTOOL)" $@ $(NVCC) -c $< $(NVCCFLAGS)


# 包含通用模块配置（如 module.am）
include $(top_srcdir)/config/module.am

endif
