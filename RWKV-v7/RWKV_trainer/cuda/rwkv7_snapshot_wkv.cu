#include <assert.h>
#include <cuda_bf16.h>

#ifdef _FP32_
using bf = float;
#define to_float(u) (u)
#define to_bf(u) (u)
#else
using bf = __nv_bfloat16;
#define to_float(u) (__bfloat162float(u))
#define to_bf(u) (__float2bfloat16_rn(u))
#endif

using i64 = long long int;
typedef bf* __restrict__ F_;

template <int N>
__global__ void forward_kernel(
    int T,
    int H,
    F_ r_,
    F_ w_log_,
    F_ k_,
    F_ v_,
    F_ a_,
    F_ b_,
    bf* __restrict__ y_,
    float* s__,
    float* __restrict__ sa_) {
    const int bb = blockIdx.y;
    const int hh = blockIdx.x;
    const int i = threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb * H + hh) * i64((T / _CHUNK_LEN_) * N * N);
    float state[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
        state[j] = 0.0f;
    }

    __shared__ float r[_CHUNK_LEN_][N];
    __shared__ float w[_CHUNK_LEN_][N];
    __shared__ float k[_CHUNK_LEN_][N];
    __shared__ float v[_CHUNK_LEN_][N];
    __shared__ float a[_CHUNK_LEN_][N];
    __shared__ float b[_CHUNK_LEN_][N];

    for (int t0 = 0; t0 < T; t0 += _CHUNK_LEN_) {
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < _CHUNK_LEN_; ++tt) {
            const int idx = ((bb * T + t0 + tt) * H + hh) * N + i;
            r[tt][i] = to_float(r_[idx]);
            // The snapshot already supplies w_log = -exp(-0.5) * sigmoid(raw_w).
            w[tt][i] = __expf(to_float(w_log_[idx]));
            k[tt][i] = to_float(k_[idx]);
            v[tt][i] = to_float(v_[idx]);
            a[tt][i] = to_float(a_[idx]);
            b[tt][i] = to_float(b_[idx]);
        }
        __syncthreads();

        for (int tt = 0; tt < _CHUNK_LEN_; ++tt) {
            const int idx = ((bb * T + t0 + tt) * H + hh) * N + i;
            float sa = 0.0f;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                sa += state[j] * a[tt][j];
            }
            sa_[idx] = sa;

            const float vi = v[tt][i];
            float y = 0.0f;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const float s = state[j] * w[tt][j] + sa * b[tt][j] + k[tt][j] * vi;
                state[j] = s;
                y += s * r[tt][j];
            }
            y_[idx] = to_bf(y);
        }

        const int base = (t0 / _CHUNK_LEN_) * N * N + i;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            s_[base + j * N] = state[j];
        }
    }
}

template <int N, int TILE>
__global__ void backward_kernel(
    int T,
    int H,
    F_ r_,
    F_ w_log_,
    F_ k_,
    F_ v_,
    F_ a_,
    F_ b_,
    F_ dy_,
    float* __restrict__ s__,
    float* __restrict__ sa_,
    bf* dr_,
    bf* dw_log_,
    bf* dk_,
    bf* dv_,
    bf* da_,
    bf* db_) {
    const int bb = blockIdx.y;
    const int hh = blockIdx.x;
    const int i = threadIdx.x;
    float* __restrict__ s_ = s__ + i64(bb * H + hh) * i64((T / _CHUNK_LEN_) * N * N);

    float stateT[N] = {0}, dstate[N] = {0}, dstateT[N] = {0};
    static_assert(_CHUNK_LEN_ % TILE == 0, "TILE must divide _CHUNK_LEN_");
    __shared__ float r[TILE][N];
    __shared__ float w[TILE][N];
    __shared__ float k[TILE][N];
    __shared__ float v[TILE][N];
    __shared__ float a[TILE][N];
    __shared__ float b[TILE][N];
    __shared__ float dy[TILE][N];
    __shared__ float sa[TILE][N];
    __shared__ float dSb_shared[N];
    float ri, wi, ki, ai, bi, dyi;

    for (int t0 = T - _CHUNK_LEN_; t0 >= 0; t0 -= _CHUNK_LEN_) {
        const int base = (t0 / _CHUNK_LEN_) * N * N + i * N;
        const float4* s4 = (const float4*)(s_ + base);
#pragma unroll
        for (int j4 = 0; j4 < N / 4; ++j4) {
            const float4 q = s4[j4];
            const int j = j4 << 2;
            stateT[j + 0] = q.x;
            stateT[j + 1] = q.y;
            stateT[j + 2] = q.z;
            stateT[j + 3] = q.w;
        }

        for (int subt = _CHUNK_LEN_ - TILE; subt >= 0; subt -= TILE) {
            __syncthreads();
#pragma unroll
            for (int tt = 0; tt < TILE; ++tt) {
                const int idx = bb * T * H * N + (t0 + subt + tt) * H * N + hh * N + i;
                r[tt][i] = to_float(r_[idx]);
                w[tt][i] = __expf(to_float(w_log_[idx]));
                k[tt][i] = to_float(k_[idx]);
                v[tt][i] = to_float(v_[idx]);
                a[tt][i] = to_float(a_[idx]);
                b[tt][i] = to_float(b_[idx]);
                dy[tt][i] = to_float(dy_[idx]);
                sa[tt][i] = sa_[idx];
            }
            __syncthreads();

            for (int tt = TILE - 1; tt >= 0; --tt) {
                const int idx = bb * T * H * N + (t0 + subt + tt) * H * N + hh * N + i;
                ri = r[tt][i];
                wi = w[tt][i];
                ki = k[tt][i];
                ai = a[tt][i];
                bi = b[tt][i];
                dyi = dy[tt][i];

                float dr = 0.0f;
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    dr += stateT[j] * dy[tt][j];
                }
                dr_[idx] = to_bf(dr);

                const float iwi = 1.0f / wi;
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    stateT[j] = (stateT[j] - ki * v[tt][j] - bi * sa[tt][j]) * iwi;
                    dstate[j] += dyi * r[tt][j];
                    dstateT[j] += ri * dy[tt][j];
                }

                float dw_log = 0.0f, dk = 0.0f, dv = 0.0f, db = 0.0f, dSb = 0.0f;
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    dw_log += dstateT[j] * stateT[j];
                    dk += dstateT[j] * v[tt][j];
                    dv += dstate[j] * k[tt][j];
                    dSb += dstate[j] * b[tt][j];
                    db += dstateT[j] * sa[tt][j];
                }
                dw_log_[idx] = to_bf(dw_log * wi);
                dk_[idx] = to_bf(dk);
                dv_[idx] = to_bf(dv);
                db_[idx] = to_bf(db);

                __syncthreads();
                dSb_shared[i] = dSb;
                __syncthreads();

                float da = 0.0f;
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    da += stateT[j] * dSb_shared[j];
                }
                da_[idx] = to_bf(da);

#pragma unroll
                for (int j = 0; j < N; ++j) {
                    dstate[j] = dstate[j] * w[tt][j] + dSb * a[tt][j];
                    dstateT[j] = dstateT[j] * iwi + ai * dSb_shared[j];
                }
            }
        }
    }
}

void cuda_forward_snapshot(int B, int T, int H, bf* r, bf* w_log, bf* k, bf* v, bf* a, bf* b, bf* y, float* s, float* sa) {
    forward_kernel<_N_><<<dim3(H, B), dim3(_N_)>>>(T, H, r, w_log, k, v, a, b, y, s, sa);
}

void cuda_backward_snapshot(int B, int T, int H, bf* r, bf* w_log, bf* k, bf* v, bf* a, bf* b, bf* dy, float* s, float* sa, bf* dr, bf* dw_log, bf* dk, bf* dv, bf* da, bf* db) {
    assert(T % _CHUNK_LEN_ == 0);
    backward_kernel<_N_, 16><<<dim3(H, B), dim3(_N_)>>>(T, H, r, w_log, k, v, a, b, dy, s, sa, dr, dw_log, dk, dv, da, db);
}
