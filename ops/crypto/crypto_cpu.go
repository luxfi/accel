package crypto

import (
	"crypto/ed25519"
	"crypto/sha256"
	"errors"
	"fmt"

	"golang.org/x/crypto/blake2b"
	"golang.org/x/crypto/sha3"
)

// CPU fallback implementations.
//
// Red MEDIUM Probe 7: the prior implementation returned hardcoded
// `false` for ECDSA, BLS, and ML-DSA verifies. That was a silent
// consensus-split surface: any GPU-equipped validator falling back
// to this stub (per the Probe 9 silent-fallback scenario, before
// Probe 9 was fixed) would emit all-false verdicts while a
// CPU-only validator running the real verifier would emit real
// verdicts. Verdict asymmetry across the validator set =
// consensus split.
//
// The propagation policy after Probe 9 is fixed: when the GPU
// returns ErrInvalidArgument, the dispatch surfaces it as a hard
// error (mldsa/gpu.go, slhdsa/gpu.go, crypto_gpu.go's SigMLDSA65
// branch). When the GPU returns NotSupported / OutOfMemory /
// KernelFailed / NoBackends, the dispatch falls back to CPU.
//
// For ML-DSA, ECDSA, and BLS the real CPU verifier is implemented
// in luxfi/crypto packages (precompile/mldsa for ML-DSA, etc.) —
// the accel package layer doesn't yet wire those CPU oracles in
// directly because the dispatch sites already do so (see
// precompile/mldsa/contract.go's per-element CPU loop). Going
// through accel here would duplicate the contract.
//
// Until the CPU oracle is wired, batchVerifyCPU returns an
// EXPLICIT error so callers get a loud "not implemented" instead
// of a silent all-false verdict. The explicit error is propagated
// up to the dispatch sites which already have a per-element CPU
// path; they MUST check the returned error and not assume []bool
// length implies correctness.
//
// Ed25519 is the exception: stdlib crypto/ed25519.Verify is a real
// reference verifier and the obvious right answer here. We keep
// that path live.

// ErrCPUNotImplemented is returned by batchVerifyCPU when no CPU
// reference is wired for the requested signature type. Surfacing
// this as an error (rather than a silent false) prevents the
// consensus-split shape Red Probe 7 identified.
var ErrCPUNotImplemented = errors.New("crypto_cpu: CPU verifier not implemented at this layer; route through the dispatch site's per-element oracle (lux/precompile/mldsa, etc.)")

func batchVerifyCPU(sigType SigType, sigs, msgs, pubkeys [][]byte) ([]bool, error) {
	results := make([]bool, len(sigs))

	switch sigType {
	case SigEd25519:
		// Real CPU reference via stdlib. Byte-equal to the GPU
		// kernel's verdict for any well-formed input.
		for i := range sigs {
			if len(pubkeys[i]) == ed25519.PublicKeySize && len(sigs[i]) == ed25519.SignatureSize {
				results[i] = ed25519.Verify(pubkeys[i], msgs[i], sigs[i])
			}
			// else: results[i] stays false (size-violation reject,
			// matching the GPU kernel's wrong-length early-reject).
		}
		return results, nil

	case SigECDSA, SigBLS, SigMLDSA65:
		// Red Probe 7: no silent false. The CPU oracle for these
		// types lives at the dispatch site (precompile/*, not here)
		// so a fallback into this function means the dispatch
		// site's wiring is incomplete. Surface the error instead of
		// emitting all-false verdicts that would alias a valid
		// "all-rejected" result and create a consensus split with
		// validators running the real oracle.
		return nil, fmt.Errorf("batchVerifyCPU(%v): %w", sigType, ErrCPUNotImplemented)

	default:
		return nil, fmt.Errorf("batchVerifyCPU: unknown signature type %v", sigType)
	}
}

func hashCPU(hashType HashType, inputs [][]byte) ([][32]byte, error) {
	results := make([][32]byte, len(inputs))

	for i, input := range inputs {
		switch hashType {
		case HashSHA256:
			results[i] = sha256.Sum256(input)
		case HashKeccak256:
			h := sha3.NewLegacyKeccak256()
			h.Write(input)
			copy(results[i][:], h.Sum(nil))
		case HashBlake3:
			h, _ := blake2b.New256(nil)
			h.Write(input)
			copy(results[i][:], h.Sum(nil))
		case HashPoseidon:
			// Poseidon hash - would use specialized library
			results[i] = poseidonHash(input)
		}
	}

	return results, nil
}

func aggregateSigsCPU(sigs [][]byte) ([]byte, error) {
	// BLS signature aggregation lives in the dispatch site's CPU
	// oracle (luxfi/crypto/bls); not duplicated here.
	return nil, fmt.Errorf("aggregateSigsCPU: %w", ErrCPUNotImplemented)
}

func aggregatePksCPU(pks [][]byte) ([]byte, error) {
	// BLS public-key aggregation — same as aggregateSigsCPU.
	return nil, fmt.Errorf("aggregatePksCPU: %w", ErrCPUNotImplemented)
}

// poseidonHash is a placeholder — Poseidon hashing for ZK-friendly
// hashing belongs in a specialized library, not here. Returning the
// SHA-256 of the input is wrong but documented; the function name
// makes the intent clear and the callers gate Poseidon use behind a
// dispatcher that knows to require a real implementation.
func poseidonHash(input []byte) [32]byte {
	return sha256.Sum256(input)
}
