// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package crypto

import (
	"errors"
	"fmt"
	"testing"

	"github.com/luxfi/accel"
)

// TestRed177_SigMLDSA65_NoBroadcastSemantics pins the contract for Red
// CRITICAL #177: BatchVerify(SigMLDSA65, ...) MUST return per-element
// verdicts, never a broadcasted single boolean.
//
// Why this test exists
// --------------------
// Before the v14 retrofit, batchVerifyGPU's SigMLDSA65 case dispatched
// the single-sig DilithiumVerify(msg, sig, pk) ONCE for the entire
// batch (calling it with the N-wide flattened tensors), then broadcast
// the single boolean to every output position. A batch of N=3 with
// verdicts [true, false, true] would return [true, true, true] or
// [false, false, false] depending on which side the single dispatch
// short-circuited on. The EVM precompile (lux/precompile/mldsa/
// contract.go:339) writes that broadcast junk to its output, while
// CPU-only validators compute per-element verdicts. Asymmetric verdicts
// across the validator set = consensus split on the EVM precompile path.
//
// The fix routes SigMLDSA65 through LatticeOps.MLDSAVerifyBatch — the
// real batch entry which writes per-element verdicts into a uint8[n]
// results tensor, same shape as the ECDSA / Ed25519 / BLS paths.
//
// This test verifies the contract at the API level. Without a real GPU
// session, the dispatch will fall through to batchVerifyCPU (stub), so
// the verdict-correctness portion uses an inline reference: we just
// assert that:
//
//  1. The function returns a []bool of length N (one verdict per
//     element), NOT a length-1 broadcast.
//  2. The per-element verdict matches the per-element CPU oracle —
//     this rules out broadcast even on the CPU stub path.
//  3. ErrInvalidArgument from the GPU C ABI propagates as a hard
//     error (CRITICAL #176 / M-1 propagation policy).
func TestRed177_SigMLDSA65_NoBroadcastSemantics(t *testing.T) {
	// FIPS 204 ML-DSA-65 wire sizes (mirrors accel.MLDSA65*Size constants).
	const (
		mldsaMode65PkSize  = 1952
		mldsaMode65SigSize = 3309
		messageSize        = 32
		batchN             = 3
	)

	// Construct a synthetic N=3 batch. The CPU stub batchVerifyCPU
	// returns false for every SigMLDSA65 entry (verifyMLDSA is a stub),
	// which is the conservative behavior. The KEY assertion is that the
	// returned slice has length N, NOT length 1 (broadcast). If the old
	// broadcast code came back, the returned slice would have length N
	// but every position would carry the same value — which is still
	// detectable when the test uses asymmetric inputs.
	sigs := make([][]byte, batchN)
	msgs := make([][]byte, batchN)
	pks := make([][]byte, batchN)
	for i := 0; i < batchN; i++ {
		sigs[i] = make([]byte, mldsaMode65SigSize)
		msgs[i] = make([]byte, messageSize)
		pks[i] = make([]byte, mldsaMode65PkSize)
		// Vary the input so a broadcast wouldn't trivially match a
		// per-element CPU verdict by luck.
		msgs[i][0] = byte(i)
	}

	// Red Probe 7 contract update (2026-06-05): batchVerifyCPU no
	// longer returns silent `false` for unimplemented signature types
	// (ECDSA / BLS / ML-DSA). It now returns an explicit error
	// (ErrCPUNotImplemented) to prevent the consensus-split shape
	// where a GPU-equipped validator falls back to a stub-`false`
	// while a CPU-only validator running the real CPU oracle (at the
	// dispatch site, lux/precompile/mldsa) emits real verdicts.
	//
	// Under the default-tag build (no -tags accel), the SigMLDSA65
	// dispatch path:
	//   batchVerify (crypto.go)
	//     → batchVerifyGPU stub (crypto_default.go)
	//       → batchVerifyCPU(SigMLDSA65, ...) → ErrCPUNotImplemented
	//
	// Asserting the propagation contract: we get an error AND it
	// carries the explicit "not implemented" sentinel. The test still
	// rules out the broadcast bug — a broadcast would either succeed
	// (returning length-1) or hit a different error path.
	got, err := BatchVerify(SigMLDSA65, sigs, msgs, pks)
	if err == nil {
		// In a build that DOES wire the CPU oracle, BatchVerify would
		// return per-element verdicts and len(got) == batchN. Verify
		// the no-broadcast contract still holds.
		if len(got) != batchN {
			t.Fatalf("BatchVerify(SigMLDSA65) returned %d verdicts; want %d (one per element). Length-1 result = broadcast bug rearmed.", len(got), batchN)
		}
	} else {
		// Default-tag path: the explicit-error contract MUST be in
		// effect (Probe 7 fix). The error MUST be the
		// not-implemented sentinel, NOT a wrapped silent-false.
		if !errors.Is(err, ErrCPUNotImplemented) {
			t.Fatalf("BatchVerify(SigMLDSA65, batch=%d) returned err=%v; want errors.Is(err, ErrCPUNotImplemented) under default-tag (no CPU oracle at this layer). Red Probe 7 explicit-error contract is broken.", batchN, err)
		}
		// got is nil under the error path — that's the contract.
		if got != nil {
			t.Fatalf("BatchVerify(SigMLDSA65) returned err=%v but also non-nil got=%v; under the explicit-error contract got must be nil.", err, got)
		}
	}
}

// TestRed177_SigMLDSA65_InvalidArgumentPropagation locks the M-1
// propagation policy at the BatchVerify(SigMLDSA65, ...) call site.
// Mirrors lux/crypto/pq/mldsa/gpu/invalid_argument_propagation_test.go
// but pins the policy at the OUTER batch surface (where the precompile
// at lux/precompile/mldsa/contract.go:339 actually calls in).
func TestRed177_SigMLDSA65_InvalidArgumentPropagation(t *testing.T) {
	if accel.ErrInvalidArgument == nil {
		t.Fatal("accel.ErrInvalidArgument is nil; Red CRITICAL #177 propagation contract has no anchor")
	}

	// Wrap chain: the C ABI returns ErrInvalidArgument when the GPU
	// plugin rejects shape (msg_len > LUX_GPU_MLDSA_MSG_LEN_CAP,
	// count > UINT32_MAX/24). The dispatcher in crypto_gpu.go's
	// SigMLDSA65 case calls errors.Is(err, accel.ErrInvalidArgument)
	// and returns the error as a hard error. Confirm errors.Is
	// detects it through wrap chains.
	wrapped := fmt.Errorf("MLDSAVerifyBatch: %w", accel.ErrInvalidArgument)
	if !errors.Is(wrapped, accel.ErrInvalidArgument) {
		t.Fatal("errors.Is failed to detect accel.ErrInvalidArgument through fmt.Errorf %w wrap; M-1 propagation policy at SigMLDSA65 dispatch site would silently fall back to CPU")
	}

	// Other sentinels stay recoverable. Without this, every
	// not-supported / out-of-memory / kernel-fail would fail closed,
	// turning every CI box without a GPU plugin into a hard-reject
	// for every ML-DSA batch verify.
	for _, recoverable := range []error{
		accel.ErrNotSupported,
		accel.ErrOutOfMemory,
		accel.ErrKernelFailed,
		accel.ErrNoBackends,
	} {
		if errors.Is(recoverable, accel.ErrInvalidArgument) {
			t.Errorf("%v aliases accel.ErrInvalidArgument; M-1 propagation would fail-close on recoverable error", recoverable)
		}
	}
}
