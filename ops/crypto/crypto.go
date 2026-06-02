// Package crypto provides GPU-accelerated cryptographic operations.
//
// All operations use the unified accel backend selection and fallback to CPU.
// Backend is selected at runtime via GPU_BACKEND env or auto-detection.
package crypto

import (
	"errors"

	"github.com/luxfi/accel"
)

// SigType identifies the signature algorithm.
type SigType uint8

const (
	SigECDSA   SigType = iota // secp256k1
	SigEd25519                // Ed25519
	SigBLS                    // BLS12-381
	SigMLDSA65                // Post-quantum ML-DSA-65
)

// HashType identifies the hash algorithm.
type HashType uint8

const (
	HashSHA256 HashType = iota
	HashKeccak256
	HashBlake3
	HashPoseidon
)

// Curve identifies the elliptic curve for MSM (multi-scalar multiplication).
//
// Wire encoding for MSM(curve, scalars, points):
//
//	secp256k1 / bn254 g1 / banderwagon : 32-byte BE x || 32-byte BE y per point (64 bytes total)
//	bls12_381 g1                       : 48-byte BE x || 48-byte BE y per point (96 bytes total)
//	scalars                            : 32-byte LE canonical (all curves)
//
// Identity (point at infinity) is wire-encoded as all-zero point bytes.
//
// The numeric values match GPUKIT_CURVE_* in luxcpp/crypto so the cgo bridge
// passes them through unchanged.
type Curve uint32

const (
	CurveSecp256k1   Curve = 0
	CurveBN254       Curve = 1 // BN254 G1
	CurveBLS12_381   Curve = 2 // BLS12-381 G1
	CurveBanderwagon Curve = 3
)

// pointSize returns the wire-encoded size (in bytes) of a single affine
// point on the given curve.
func (c Curve) pointSize() int {
	switch c {
	case CurveBLS12_381:
		return 96
	default:
		return 64
	}
}

var (
	ErrInvalidInput = errors.New("crypto: invalid input")
	ErrBatchEmpty   = errors.New("crypto: empty batch")
	ErrVerifyFailed = errors.New("crypto: verification failed")
	ErrUnsupported  = errors.New("crypto: unsupported curve")
)

// BatchVerify verifies multiple signatures in parallel.
// Returns a slice of bools indicating which signatures are valid.
// Uses GPU acceleration when available, falls back to CPU.
func BatchVerify(sigType SigType, sigs, msgs, pubkeys [][]byte) ([]bool, error) {
	if len(sigs) == 0 || len(sigs) != len(msgs) || len(sigs) != len(pubkeys) {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return batchVerifyCPU(sigType, sigs, msgs, pubkeys)
	}
	defer sess.Close()

	return batchVerifyGPU(sess, sigType, sigs, msgs, pubkeys)
}

// Hash computes hashes for multiple inputs in parallel.
func Hash(hashType HashType, inputs [][]byte) ([][32]byte, error) {
	if len(inputs) == 0 {
		return nil, ErrBatchEmpty
	}

	sess, err := accel.NewSession()
	if err != nil {
		return hashCPU(hashType, inputs)
	}
	defer sess.Close()

	return hashGPU(sess, hashType, inputs)
}

// MSM performs multi-scalar multiplication on the given curve.
//
//	result = sum_{i=0..n-1}  scalars[i] * points[i]
//
// scalars[i] is 32-byte LE canonical; points[i] is the curve's wire-encoded
// affine form (see Curve doc). The result is the same wire format as a
// single point on the curve.
//
// Routes through the GPU when available; falls back to a CPU implementation
// (luxcpp/crypto via cgo when built with -tags=lux_crypto_native, otherwise
// a pure-Go gnark-crypto path).
func MSM(curve Curve, scalars, points [][]byte) ([]byte, error) {
	if len(scalars) != len(points) || len(scalars) == 0 {
		return nil, ErrInvalidInput
	}

	sess, err := accel.NewSession()
	if err != nil {
		return msmCPU(curve, scalars, points)
	}
	defer sess.Close()

	return msmGPU(sess, curve, scalars, points)
}

// AggregateSignatures aggregates BLS signatures.
func AggregateSignatures(sigs [][]byte) ([]byte, error) {
	if len(sigs) == 0 {
		return nil, ErrBatchEmpty
	}

	sess, err := accel.NewSession()
	if err != nil {
		return aggregateSigsCPU(sigs)
	}
	defer sess.Close()

	return aggregateSigsGPU(sess, sigs)
}

// AggregatePublicKeys aggregates BLS public keys.
func AggregatePublicKeys(pks [][]byte) ([]byte, error) {
	if len(pks) == 0 {
		return nil, ErrBatchEmpty
	}

	sess, err := accel.NewSession()
	if err != nil {
		return aggregatePksCPU(pks)
	}
	defer sess.Close()

	return aggregatePksGPU(sess, pks)
}
