//go:build accel

package fhe

import (
	"github.com/luxfi/accel"
)

// GPU-accelerated FHE implementations.
// Uses kernels from fhe/ kernel collection.

func keyGenGPU(sess *accel.Session, params Params) (*SecretKey, *PublicKey, error) {
	// Uses fhe/keygen kernel for parallel key generation
	// Kernel: fhe/bfv_keygen or fhe/ckks_keygen based on scheme
	//
	// GPU acceleration provides:
	// - Parallel polynomial sampling
	// - Parallel NTT for key computation
	// - Batch RNS operations
	//
	// TODO: Implement kernel dispatch via Session
	return keyGenCPU(params)
}

func genRelinKeyGPU(sess *accel.Session, params Params, sk *SecretKey) (*RelinKey, error) {
	// Uses fhe/relin_keygen kernel
	// Generates evaluation key for s^2 -> s conversion
	//
	// GPU acceleration provides:
	// - Parallel gadget decomposition
	// - Batch NTT transforms
	// - Parallel encryption of decomposed components
	//
	// TODO: Implement kernel dispatch
	return genRelinKeyCPU(params, sk)
}

func genGaloisKeyGPU(sess *accel.Session, params Params, sk *SecretKey, steps []int) (*GaloisKey, error) {
	// Uses fhe/galois_keygen kernel
	// Generates keys for rotation automorphisms
	//
	// GPU acceleration provides:
	// - Parallel key generation for multiple steps
	// - Batch automorphism evaluation
	//
	// TODO: Implement kernel dispatch
	return genGaloisKeyCPU(params, sk, steps)
}

func genBootstrapKeyGPU(sess *accel.Session, params Params, sk *SecretKey) (*BootstrapKey, error) {
	// Uses fhe/bootstrap_keygen kernel
	// Generates large evaluation key for bootstrapping
	//
	// GPU acceleration is critical for bootstrap key generation
	// due to the size and complexity of the key material
	//
	// TODO: Implement kernel dispatch
	return genBootstrapKeyCPU(params, sk)
}

func encryptGPU(sess *accel.Session, params Params, pk *PublicKey, pt *Plaintext) (*Ciphertext, error) {
	// Uses fhe/encrypt kernel
	// Kernel: fhe/bfv_encrypt or fhe/ckks_encrypt based on scheme
	//
	// GPU acceleration provides:
	// - Parallel polynomial multiplication via NTT
	// - Parallel error sampling
	// - Batch RNS operations
	//
	// TODO: Implement kernel dispatch
	return encryptCPU(params, pk, pt)
}

func decryptGPU(sess *accel.Session, params Params, sk *SecretKey, ct *Ciphertext) (*Plaintext, error) {
	// Uses fhe/decrypt kernel
	// Kernel: fhe/bfv_decrypt or fhe/ckks_decrypt based on scheme
	//
	// GPU acceleration provides:
	// - Parallel inner product computation
	// - Fast NTT/inverse NTT
	//
	// TODO: Implement kernel dispatch
	return decryptCPU(params, sk, ct)
}

func addGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	// Uses fhe/add kernel
	// Simple component-wise addition in RNS domain
	//
	// GPU acceleration provides:
	// - Parallel modular addition across all coefficients
	// - Batch processing of RNS components
	//
	// TODO: Implement kernel dispatch
	return addCPU(params, ct1, ct2)
}

func addPlainGPU(sess *accel.Session, params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	// Uses fhe/add_plain kernel
	//
	// TODO: Implement kernel dispatch
	return addPlainCPU(params, ct, pt)
}

func subGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	// Uses fhe/sub kernel
	//
	// TODO: Implement kernel dispatch
	return subCPU(params, ct1, ct2)
}

func multiplyGPU(sess *accel.Session, params Params, ct1, ct2 *Ciphertext, rlk *RelinKey) (*Ciphertext, error) {
	// Uses fhe/multiply kernel followed by fhe/relinearize
	//
	// GPU acceleration provides:
	// - Parallel NTT transforms for polynomial multiplication
	// - Parallel tensor product computation
	// - Efficient relinearization via key switching
	//
	// This is the most computationally intensive operation and
	// benefits greatly from GPU acceleration.
	//
	// TODO: Implement kernel dispatch
	return multiplyCPU(params, ct1, ct2, rlk)
}

func multiplyPlainGPU(sess *accel.Session, params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	// Uses fhe/multiply_plain kernel
	// Simpler than ct-ct multiplication (no relinearization needed)
	//
	// TODO: Implement kernel dispatch
	return multiplyPlainCPU(params, ct, pt)
}

func rotateGPU(sess *accel.Session, params Params, ct *Ciphertext, gk *GaloisKey, steps int) (*Ciphertext, error) {
	// Uses fhe/rotate kernel
	//
	// GPU acceleration provides:
	// - Parallel Galois automorphism application
	// - Efficient key switching
	//
	// TODO: Implement kernel dispatch
	return rotateCPU(params, ct, gk, steps)
}

func rescaleGPU(sess *accel.Session, params Params, ct *Ciphertext) (*Ciphertext, error) {
	// Uses fhe/rescale kernel (CKKS only)
	//
	// GPU acceleration provides:
	// - Parallel RNS base conversion
	// - Efficient modular reduction
	//
	// TODO: Implement kernel dispatch
	return rescaleCPU(params, ct)
}

func modSwitchGPU(sess *accel.Session, params Params, ct *Ciphertext) (*Ciphertext, error) {
	// Uses fhe/modswitch kernel
	//
	// TODO: Implement kernel dispatch
	return modSwitchCPU(params, ct)
}

func bootstrapGPU(sess *accel.Session, params Params, ct *Ciphertext, bsk *BootstrapKey) (*Ciphertext, error) {
	// Uses fhe/bootstrap kernel
	//
	// Bootstrapping is the most complex FHE operation, consisting of:
	// 1. Modulus raising
	// 2. Homomorphic decryption (slot-to-coeff + modular reduction)
	// 3. Re-encryption
	//
	// GPU acceleration is essential for practical bootstrapping
	// as it requires many polynomial multiplications and rotations.
	//
	// TODO: Implement kernel dispatch
	return bootstrapCPU(params, ct, bsk)
}

func batchEncryptGPU(sess *accel.Session, params Params, pk *PublicKey, pts []*Plaintext) ([]*Ciphertext, error) {
	// Uses fhe/batch_encrypt kernel
	// Encrypts multiple plaintexts in parallel
	//
	// GPU acceleration provides:
	// - Full parallelization across all plaintexts
	// - Shared polynomial operations (NTT, sampling)
	//
	// TODO: Implement kernel dispatch
	return batchEncryptCPU(params, pk, pts)
}

func batchDecryptGPU(sess *accel.Session, params Params, sk *SecretKey, cts []*Ciphertext) ([]*Plaintext, error) {
	// Uses fhe/batch_decrypt kernel
	// Decrypts multiple ciphertexts in parallel
	//
	// TODO: Implement kernel dispatch
	return batchDecryptCPU(params, sk, cts)
}
