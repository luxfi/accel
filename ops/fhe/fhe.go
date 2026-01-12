// Package fhe provides GPU-accelerated Fully Homomorphic Encryption operations.
//
// Supports BFV (exact integer arithmetic) and CKKS (approximate real arithmetic) schemes.
// All operations use the unified accel backend selection and fallback to CPU.
// Backend is selected at runtime via LUX_ACCEL_BACKEND env or auto-detection.
package fhe

import (
	"errors"

	"github.com/luxfi/accel"
)

// Scheme identifies the FHE scheme.
type Scheme uint8

const (
	SchemeBFV  Scheme = iota // BFV: exact integer arithmetic
	SchemeCKKS               // CKKS: approximate real arithmetic
)

// Params contains FHE scheme parameters.
type Params struct {
	Scheme     Scheme   // BFV or CKKS
	PolyDegree uint32   // N: polynomial degree (power of 2, e.g., 4096, 8192, 16384)
	CoeffMods  []uint64 // q: coefficient modulus chain
	PlainMod   uint64   // t: plaintext modulus (BFV only)
	Scale      float64  // Delta: encoding scale (CKKS only)
}

// Ciphertext represents an encrypted value.
type Ciphertext struct {
	Data   []uint64 // Polynomial coefficients in RNS form
	Scheme Scheme   // BFV or CKKS
	Level  int      // Current modulus level
	Scale  float64  // Current scale (CKKS only)
}

// Plaintext represents a plaintext value.
type Plaintext struct {
	Data   []uint64 // Encoded polynomial
	Scheme Scheme   // BFV or CKKS
	Scale  float64  // Encoding scale (CKKS only)
}

// SecretKey for decryption.
type SecretKey struct {
	Data []uint64
}

// PublicKey for encryption.
type PublicKey struct {
	Data []uint64
}

// RelinKey for relinearization after multiplication.
type RelinKey struct {
	Data []uint64
}

// GaloisKey for rotation operations.
type GaloisKey struct {
	Data  []uint64
	Steps []int // Rotation steps this key supports
}

// BootstrapKey for refreshing ciphertext noise.
type BootstrapKey struct {
	Data []uint64
}

var (
	ErrInvalidParams   = errors.New("fhe: invalid parameters")
	ErrInvalidInput    = errors.New("fhe: invalid input")
	ErrKeyMismatch     = errors.New("fhe: key does not match ciphertext")
	ErrLevelExhausted  = errors.New("fhe: modulus levels exhausted")
	ErrScaleMismatch   = errors.New("fhe: scale mismatch")
	ErrRotationKey     = errors.New("fhe: rotation key not available for requested steps")
	ErrBootstrapFailed = errors.New("fhe: bootstrap operation failed")
)

// KeyGen generates a secret/public key pair.
func KeyGen(params Params) (*SecretKey, *PublicKey, error) {
	if err := validateParams(params); err != nil {
		return nil, nil, err
	}

	sess, err := accel.NewSession()
	if err != nil {
		return keyGenCPU(params)
	}
	defer sess.Close()

	return keyGenGPU(sess, params)
}

// GenRelinKey generates a relinearization key.
func GenRelinKey(params Params, sk *SecretKey) (*RelinKey, error) {
	if sk == nil || len(sk.Data) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return genRelinKeyCPU(params, sk)
	}
	defer sess.Close()

	return genRelinKeyGPU(sess, params, sk)
}

// GenGaloisKey generates Galois keys for rotation.
func GenGaloisKey(params Params, sk *SecretKey, steps []int) (*GaloisKey, error) {
	if sk == nil || len(sk.Data) == 0 || len(steps) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return genGaloisKeyCPU(params, sk, steps)
	}
	defer sess.Close()

	return genGaloisKeyGPU(sess, params, sk, steps)
}

// GenBootstrapKey generates a bootstrapping key.
func GenBootstrapKey(params Params, sk *SecretKey) (*BootstrapKey, error) {
	if sk == nil || len(sk.Data) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return genBootstrapKeyCPU(params, sk)
	}
	defer sess.Close()

	return genBootstrapKeyGPU(sess, params, sk)
}

// Encrypt encrypts a plaintext.
func Encrypt(params Params, pk *PublicKey, pt *Plaintext) (*Ciphertext, error) {
	if pk == nil || pt == nil {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return encryptCPU(params, pk, pt)
	}
	defer sess.Close()

	return encryptGPU(sess, params, pk, pt)
}

// Decrypt decrypts a ciphertext.
func Decrypt(params Params, sk *SecretKey, ct *Ciphertext) (*Plaintext, error) {
	if sk == nil || ct == nil {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return decryptCPU(params, sk, ct)
	}
	defer sess.Close()

	return decryptGPU(sess, params, sk, ct)
}

// Add performs homomorphic addition of two ciphertexts.
func Add(params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	if ct1 == nil || ct2 == nil {
		return nil, ErrInvalidInput
	}
	if ct1.Scheme != ct2.Scheme {
		return nil, ErrInvalidInput
	}
	if ct1.Scheme == SchemeCKKS && ct1.Scale != ct2.Scale {
		return nil, ErrScaleMismatch
	}

	sess, err := accel.NewSession()
	if err != nil {
		return addCPU(params, ct1, ct2)
	}
	defer sess.Close()

	return addGPU(sess, params, ct1, ct2)
}

// AddPlain adds a plaintext to a ciphertext.
func AddPlain(params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	if ct == nil || pt == nil {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return addPlainCPU(params, ct, pt)
	}
	defer sess.Close()

	return addPlainGPU(sess, params, ct, pt)
}

// Sub performs homomorphic subtraction of two ciphertexts.
func Sub(params Params, ct1, ct2 *Ciphertext) (*Ciphertext, error) {
	if ct1 == nil || ct2 == nil {
		return nil, ErrInvalidInput
	}
	if ct1.Scheme != ct2.Scheme {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return subCPU(params, ct1, ct2)
	}
	defer sess.Close()

	return subGPU(sess, params, ct1, ct2)
}

// Multiply performs homomorphic multiplication with relinearization.
func Multiply(params Params, ct1, ct2 *Ciphertext, rlk *RelinKey) (*Ciphertext, error) {
	if ct1 == nil || ct2 == nil || rlk == nil {
		return nil, ErrInvalidInput
	}
	if ct1.Scheme != ct2.Scheme {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return multiplyCPU(params, ct1, ct2, rlk)
	}
	defer sess.Close()

	return multiplyGPU(sess, params, ct1, ct2, rlk)
}

// MultiplyPlain multiplies a ciphertext by a plaintext.
func MultiplyPlain(params Params, ct *Ciphertext, pt *Plaintext) (*Ciphertext, error) {
	if ct == nil || pt == nil {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return multiplyPlainCPU(params, ct, pt)
	}
	defer sess.Close()

	return multiplyPlainGPU(sess, params, ct, pt)
}

// Rotate rotates ciphertext slots.
// Positive steps rotate left, negative rotate right.
func Rotate(params Params, ct *Ciphertext, gk *GaloisKey, steps int) (*Ciphertext, error) {
	if ct == nil || gk == nil {
		return nil, ErrInvalidInput
	}
	if !hasStep(gk.Steps, steps) {
		return nil, ErrRotationKey
	}

	sess, err := accel.NewSession()
	if err != nil {
		return rotateCPU(params, ct, gk, steps)
	}
	defer sess.Close()

	return rotateGPU(sess, params, ct, gk, steps)
}

// Rescale rescales a CKKS ciphertext after multiplication.
// Reduces scale and drops one modulus level.
func Rescale(params Params, ct *Ciphertext) (*Ciphertext, error) {
	if ct == nil {
		return nil, ErrInvalidInput
	}
	if ct.Scheme != SchemeCKKS {
		return nil, ErrInvalidParams
	}
	if ct.Level <= 0 {
		return nil, ErrLevelExhausted
	}

	sess, err := accel.NewSession()
	if err != nil {
		return rescaleCPU(params, ct)
	}
	defer sess.Close()

	return rescaleGPU(sess, params, ct)
}

// ModSwitch switches to a smaller modulus without rescaling.
// Used to match levels before homomorphic operations.
func ModSwitch(params Params, ct *Ciphertext) (*Ciphertext, error) {
	if ct == nil {
		return nil, ErrInvalidInput
	}
	if ct.Level <= 0 {
		return nil, ErrLevelExhausted
	}

	sess, err := accel.NewSession()
	if err != nil {
		return modSwitchCPU(params, ct)
	}
	defer sess.Close()

	return modSwitchGPU(sess, params, ct)
}

// Bootstrap refreshes a ciphertext by reducing noise.
// Requires a bootstrapping key and consumes multiple levels.
func Bootstrap(params Params, ct *Ciphertext, bsk *BootstrapKey) (*Ciphertext, error) {
	if ct == nil || bsk == nil {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return bootstrapCPU(params, ct, bsk)
	}
	defer sess.Close()

	return bootstrapGPU(sess, params, ct, bsk)
}

// BatchEncrypt encrypts multiple plaintexts in parallel.
func BatchEncrypt(params Params, pk *PublicKey, pts []*Plaintext) ([]*Ciphertext, error) {
	if pk == nil || len(pts) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchEncryptCPU(params, pk, pts)
	}
	defer sess.Close()

	return batchEncryptGPU(sess, params, pk, pts)
}

// BatchDecrypt decrypts multiple ciphertexts in parallel.
func BatchDecrypt(params Params, sk *SecretKey, cts []*Ciphertext) ([]*Plaintext, error) {
	if sk == nil || len(cts) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchDecryptCPU(params, sk, cts)
	}
	defer sess.Close()

	return batchDecryptGPU(sess, params, sk, cts)
}

// EncodeBFV encodes integers into a BFV plaintext.
func EncodeBFV(params Params, values []int64) (*Plaintext, error) {
	if params.Scheme != SchemeBFV {
		return nil, ErrInvalidParams
	}
	if len(values) > int(params.PolyDegree) {
		return nil, ErrInvalidInput
	}

	return encodeBFVCPU(params, values)
}

// DecodeBFV decodes a BFV plaintext to integers.
func DecodeBFV(params Params, pt *Plaintext) ([]int64, error) {
	if params.Scheme != SchemeBFV || pt == nil {
		return nil, ErrInvalidInput
	}

	return decodeBFVCPU(params, pt)
}

// EncodeCKKS encodes floats into a CKKS plaintext.
func EncodeCKKS(params Params, values []float64) (*Plaintext, error) {
	if params.Scheme != SchemeCKKS {
		return nil, ErrInvalidParams
	}
	if len(values) > int(params.PolyDegree)/2 {
		return nil, ErrInvalidInput
	}

	return encodeCKKSCPU(params, values)
}

// DecodeCKKS decodes a CKKS plaintext to floats.
func DecodeCKKS(params Params, pt *Plaintext) ([]float64, error) {
	if params.Scheme != SchemeCKKS || pt == nil {
		return nil, ErrInvalidInput
	}

	return decodeCKKSCPU(params, pt)
}

// validateParams checks that parameters are valid.
func validateParams(p Params) error {
	if p.PolyDegree == 0 || (p.PolyDegree&(p.PolyDegree-1)) != 0 {
		return ErrInvalidParams // Must be power of 2
	}
	if len(p.CoeffMods) == 0 {
		return ErrInvalidParams
	}
	if p.Scheme == SchemeBFV && p.PlainMod == 0 {
		return ErrInvalidParams
	}
	if p.Scheme == SchemeCKKS && p.Scale <= 0 {
		return ErrInvalidParams
	}
	return nil
}

// hasStep checks if the GaloisKey supports a rotation step.
func hasStep(steps []int, step int) bool {
	for _, s := range steps {
		if s == step {
			return true
		}
	}
	return false
}
