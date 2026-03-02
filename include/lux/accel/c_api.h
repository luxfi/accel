#ifndef LUX_ACCEL_C_API_H
#define LUX_ACCEL_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes */
typedef int lux_status;
#define LUX_OK              0
#define LUX_ERROR           1
#define LUX_OUT_OF_MEMORY   2
#define LUX_INVALID_ARGUMENT 3
#define LUX_NOT_SUPPORTED   4
#define LUX_NO_BACKEND      5
#define LUX_KERNEL_ERROR    6

/* Opaque handles */
typedef void* lux_session;
typedef void* lux_tensor;

/* Backend types */
typedef int lux_backend_type;

/* Data types */
typedef int lux_dtype;

/* Device info */
typedef struct {
    const char* name;
    const char* vendor;
    int backend;
    int is_discrete;
    int is_unified_memory;
    uint64_t total_memory;
    uint32_t max_workgroup_size;
    uint32_t simd_width;
} lux_device_info;

/* Library lifecycle */
lux_status lux_init(void);
void lux_shutdown(void);
const char* lux_version(void);
const char* lux_get_error(void);

/* Backend management */
lux_status lux_load_backend(const char* path);
int lux_backend_count(void);
int lux_backend_type_at(int index);
int lux_device_count(lux_backend_type backend);
lux_status lux_get_device_info(lux_backend_type backend, int index, lux_device_info* info);

/* Session management */
lux_status lux_session_create(lux_session* session);
lux_status lux_session_create_with_backend(lux_backend_type backend, lux_session* session);
lux_status lux_session_create_with_device(lux_backend_type backend, int device_index, lux_session* session);
void lux_session_destroy(lux_session session);
lux_status lux_session_sync(lux_session session);

/* Tensor management */
lux_status lux_tensor_create(lux_session session, lux_dtype dtype, const size_t* shape, size_t ndim, lux_tensor* tensor);
lux_status lux_tensor_create_with_data(lux_session session, lux_dtype dtype, const size_t* shape, size_t ndim, const void* data, size_t data_size, lux_tensor* tensor);
void lux_tensor_destroy(lux_tensor tensor);
size_t lux_tensor_ndim(lux_tensor tensor);
size_t lux_tensor_shape(lux_tensor tensor, size_t dim);
size_t lux_tensor_numel(lux_tensor tensor);
size_t lux_tensor_bytes(lux_tensor tensor);
lux_dtype lux_tensor_dtype(lux_tensor tensor);
lux_status lux_tensor_to_host(lux_tensor tensor, void* dst, size_t dst_size);
lux_status lux_tensor_from_host(lux_tensor tensor, const void* src, size_t src_size);

/* ML operations */
lux_status lux_matmul(lux_session s, lux_tensor a, lux_tensor b, lux_tensor c);
lux_status lux_relu(lux_session s, lux_tensor input, lux_tensor output);
lux_status lux_gelu(lux_session s, lux_tensor input, lux_tensor output);
lux_status lux_softmax(lux_session s, lux_tensor input, lux_tensor output, int axis);
lux_status lux_layer_norm(lux_session s, lux_tensor input, lux_tensor gamma, lux_tensor beta, lux_tensor output, float eps);
lux_status lux_attention(lux_session s, lux_tensor q, lux_tensor k, lux_tensor v, lux_tensor output, float scale);

/* Crypto operations */
lux_status lux_sha256(lux_session s, lux_tensor input, lux_tensor output);
lux_status lux_keccak256(lux_session s, lux_tensor input, lux_tensor output);
lux_status lux_poseidon(lux_session s, lux_tensor input, lux_tensor output);
lux_status lux_ecdsa_verify_batch(lux_session s, lux_tensor messages, lux_tensor signatures, lux_tensor pubkeys, lux_tensor results);
lux_status lux_ed25519_verify_batch(lux_session s, lux_tensor messages, lux_tensor signatures, lux_tensor pubkeys, lux_tensor results);
lux_status lux_bls_verify_batch(lux_session s, lux_tensor messages, lux_tensor signatures, lux_tensor pubkeys, lux_tensor results);
lux_status lux_merkle_root(lux_session s, lux_tensor leaves, lux_tensor root);

/* ZK operations */
lux_status lux_ntt(lux_session s, lux_tensor input, lux_tensor output, lux_tensor roots, uint64_t modulus);
lux_status lux_intt(lux_session s, lux_tensor input, lux_tensor output, lux_tensor inv_roots, uint64_t modulus);
lux_status lux_msm(lux_session s, lux_tensor scalars, lux_tensor bases, lux_tensor result);
lux_status lux_poly_mul(lux_session s, lux_tensor a, lux_tensor b, lux_tensor c, uint64_t modulus);

/* Lattice/PQC operations */
lux_status lux_kyber_keygen(lux_session s, lux_tensor pk, lux_tensor sk);
lux_status lux_kyber_encaps(lux_session s, lux_tensor pk, lux_tensor ct, lux_tensor ss);
lux_status lux_kyber_decaps(lux_session s, lux_tensor ct, lux_tensor sk, lux_tensor ss);
lux_status lux_dilithium_sign(lux_session s, lux_tensor msg, lux_tensor sk, lux_tensor sig);
lux_status lux_dilithium_verify(lux_session s, lux_tensor msg, lux_tensor sig, lux_tensor pk, int* valid);

/* FHE operations */
lux_status lux_bfv_encrypt(lux_session s, lux_tensor plaintext, lux_tensor pk, lux_tensor ciphertext);
lux_status lux_bfv_decrypt(lux_session s, lux_tensor ciphertext, lux_tensor sk, lux_tensor plaintext);
lux_status lux_bfv_add(lux_session s, lux_tensor ct1, lux_tensor ct2, lux_tensor result);
lux_status lux_bfv_multiply(lux_session s, lux_tensor ct1, lux_tensor ct2, lux_tensor relin_key, lux_tensor result);

/* DEX operations */
lux_status lux_constant_product_swap(lux_session s, lux_tensor reserve_x, lux_tensor reserve_y, lux_tensor amount_in, int x_to_y, lux_tensor amount_out, float fee);
lux_status lux_compute_twap(lux_session s, lux_tensor prices, lux_tensor timestamps, uint64_t start, uint64_t end, lux_tensor twap);
lux_status lux_match_orders(lux_session s, lux_tensor bids, lux_tensor asks, lux_tensor matches, lux_tensor prices, lux_tensor amounts);

#ifdef __cplusplus
}
#endif

#endif /* LUX_ACCEL_C_API_H */

