#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// The adapter validates the page table before entering this unchecked kernel.
template <typename Word>
__global__ void page_move(const Word* src, const int64_t* ids, Word* dst,
                          int64_t count, int64_t page_words, bool scatter) {
    int64_t offset = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; offset < count; offset += int64_t(blockDim.x) * gridDim.x) {
        const int64_t physical = ids[offset / page_words] * page_words
                               + offset % page_words;
        if (scatter) dst[physical] = src[offset];
        else dst[offset] = src[physical];
    }
}

void move(torch::Tensor source, torch::Tensor ids, torch::Tensor destination,
          bool scatter) {
    TORCH_CHECK(source.is_cuda() && destination.is_cuda() && ids.is_cuda(),
                "expected CUDA tensors");
    TORCH_CHECK(source.device() == destination.device() && source.device() == ids.device(),
                "expected tensors on the same CUDA device");
    TORCH_CHECK(source.is_contiguous() && destination.is_contiguous()
                && ids.is_contiguous() && ids.scalar_type() == torch::kInt64,
                "expected contiguous tensors and int64 IDs");
    c10::cuda::CUDAGuard guard(source.device());
    const int64_t page_bytes = source.numel() / source.size(0) * source.element_size();
    const int64_t count = ids.numel() * page_bytes;
    if (!count) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (page_bytes % 16 == 0 && reinterpret_cast<uintptr_t>(source.data_ptr()) % 16 == 0
        && reinterpret_cast<uintptr_t>(destination.data_ptr()) % 16 == 0) {
        const int64_t words = count / 16;
        page_move<<<std::min<int64_t>((words + 255) / 256, 65535), 256, 0, stream>>>(
            reinterpret_cast<const int4*>(source.data_ptr()), ids.data_ptr<int64_t>(),
            reinterpret_cast<int4*>(destination.data_ptr()), words, page_bytes / 16, scatter);
    } else {
        page_move<<<std::min<int64_t>((count + 255) / 256, 65535), 256, 0, stream>>>(
            static_cast<const uint8_t*>(source.data_ptr()), ids.data_ptr<int64_t>(),
            static_cast<uint8_t*>(destination.data_ptr()), count, page_bytes, scatter);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_host(torch::Tensor source, torch::Tensor ids, torch::Tensor host) {
    TORCH_CHECK(source.is_cuda() && ids.is_cuda() && !host.is_cuda() && host.is_pinned(),
                "expected CUDA source/IDs and pinned CPU output");
    c10::cuda::CUDAGuard guard(source.device());
    void* mapped = nullptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&mapped, host.data_ptr(), 0));
    const int64_t page_bytes = source.numel() / source.size(0) * source.element_size();
    const int64_t bytes = ids.numel() * page_bytes;
    TORCH_CHECK(host.nbytes() == bytes, "incorrect host output size");
    if (!bytes) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    page_move<<<std::min<int64_t>((bytes + 255) / 256, 65535), 256, 0, stream>>>(
        static_cast<const uint8_t*>(source.data_ptr()), ids.data_ptr<int64_t>(),
        static_cast<uint8_t*>(mapped), bytes, page_bytes, false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Scalar>
__global__ void quant_host(const Scalar* source, int8_t* codes, float* scales,
                          int64_t groups, int tokens, int channels) {
    const int64_t channel = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= groups * channels) return;
    const int64_t group = channel / channels;
    const int dim = channel % channels;
    float maximum = 0;
    for (int t = 0; t < tokens; ++t)
        maximum = fmaxf(maximum, fabsf(float(source[(group * tokens + t) * channels + dim])));
    const float scale = maximum == 0 ? 1 : maximum / 127;
    scales[channel] = scale;
    for (int t = 0; t < tokens; ++t) {
        const int64_t index = (group * tokens + t) * channels + dim;
        codes[index] = int8_t(fminf(127, fmaxf(-127, nearbyintf(float(source[index]) / scale))));
    }
}

void quantize_host(torch::Tensor source, torch::Tensor codes, torch::Tensor scales) {
    TORCH_CHECK(source.is_cuda() && codes.is_pinned() && scales.is_pinned(),
                "expected CUDA input and pinned output tensors");
    c10::cuda::CUDAGuard guard(source.device());
    void *mapped_codes = nullptr, *mapped_scales = nullptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_codes, codes.data_ptr(), 0));
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_scales, scales.data_ptr(), 0));
    const int tokens = source.size(-2), channels = source.size(-1);
    const int64_t groups = source.numel() / (tokens * channels);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                   source.scalar_type(), "quantize_host", [&] {
        quant_host<<<(groups * channels + 127) / 128, 128, 0, stream>>>(
            source.data_ptr<scalar_t>(), static_cast<int8_t*>(mapped_codes),
            static_cast<float*>(mapped_scales), groups, tokens, channels);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("move", &move);
    m.def("pack_host", &pack_host);
    m.def("quantize_host", &quantize_host);
}
