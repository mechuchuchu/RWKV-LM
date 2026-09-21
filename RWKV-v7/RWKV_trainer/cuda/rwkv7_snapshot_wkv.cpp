#include <torch/extension.h>
#include <cuda_bf16.h>

#include <vector>

#ifdef _FP32_
using bf = float;
#else
using bf = __nv_bfloat16;
#endif

void cuda_forward_snapshot(
    int B,
    int T,
    int H,
    bf* r,
    bf* w_log,
    bf* k,
    bf* v,
    bf* a,
    bf* b,
    bf* y,
    float* s,
    float* sa);

void cuda_backward_snapshot(
    int B,
    int T,
    int H,
    bf* r,
    bf* w_log,
    bf* k,
    bf* v,
    bf* a,
    bf* b,
    bf* dy,
    float* s,
    float* sa,
    bf* dr,
    bf* dw_log,
    bf* dk,
    bf* dv,
    bf* da,
    bf* db);

void forward(
    torch::Tensor& r,
    torch::Tensor& w_log,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& y,
    torch::Tensor& s,
    torch::Tensor& sa) {
    const int B = r.size(0);
    const int T = r.size(1);
    const int H = r.size(2);
    cuda_forward_snapshot(
        B,
        T,
        H,
        (bf*)r.data_ptr(),
        (bf*)w_log.data_ptr(),
        (bf*)k.data_ptr(),
        (bf*)v.data_ptr(),
        (bf*)a.data_ptr(),
        (bf*)b.data_ptr(),
        (bf*)y.data_ptr(),
        (float*)s.data_ptr(),
        (float*)sa.data_ptr());
}

void backward(
    torch::Tensor& r,
    torch::Tensor& w_log,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& dy,
    torch::Tensor& s,
    torch::Tensor& sa,
    torch::Tensor& dr,
    torch::Tensor& dw_log,
    torch::Tensor& dk,
    torch::Tensor& dv,
    torch::Tensor& da,
    torch::Tensor& db) {
    const int B = r.size(0);
    const int T = r.size(1);
    const int H = r.size(2);
    cuda_backward_snapshot(
        B,
        T,
        H,
        (bf*)r.data_ptr(),
        (bf*)w_log.data_ptr(),
        (bf*)k.data_ptr(),
        (bf*)v.data_ptr(),
        (bf*)a.data_ptr(),
        (bf*)b.data_ptr(),
        (bf*)dy.data_ptr(),
        (float*)s.data_ptr(),
        (float*)sa.data_ptr(),
        (bf*)dr.data_ptr(),
        (bf*)dw_log.data_ptr(),
        (bf*)dk.data_ptr(),
        (bf*)dv.data_ptr(),
        (bf*)da.data_ptr(),
        (bf*)db.data_ptr());
}

TORCH_LIBRARY(rwkv7_snapshot_wkv, m) {
    m.def("forward(Tensor r, Tensor w_log, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor r, Tensor w_log, Tensor k, Tensor v, Tensor a, Tensor b, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dr, Tensor(b!) dw_log, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) da, Tensor(f!) db) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_snapshot_wkv, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
