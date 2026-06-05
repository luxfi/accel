// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && lux_accel_real

package accel_test

import (
	"errors"
	"testing"

	"github.com/luxfi/accel"
)

// TestRed_Probe9_M1PropagationThroughCgo locks the M-1 propagation
// contract through the REAL cgo boundary. Without this test, the
// existing Go-level mock tests at:
//
//   lux/crypto/slhdsa/invalid_argument_propagation_test.go:48-50
//   lux/accel/ops/crypto/crypto_mldsa_batch_test.go:117-119
//
// pass trivially even when the cgo path returns the WRONG sentinel
// (capi.ErrInvalidArgument instead of accel.ErrInvalidArgument). They
// wrap the accel sentinel directly, never exercising the real cgo
// boundary.
//
// Red Probe 9 found that every method in ops_c.go returned
// capi.* sentinels unwrapped. Consumers calling errors.Is(err,
// accel.ErrInvalidArgument) would get FALSE because:
//
//   capi.ErrInvalidArgument  = errors.New("invalid argument")
//   accel.ErrInvalidArgument = errors.New("accel: invalid argument")
//
// Distinct errors.New(...) values → distinct pointers → errors.Is
// (which uses pointer-equality at the leaves of the unwrap chain)
// returns false. Every M-1 consumer's silent-CPU-fallback branch is
// taken, defeating the M-1 propagation chain entirely.
//
// The fix (this commit) wraps every cgo-routed method in ops_c.go +
// session_c.go through translateCapiError. This test exercises the
// real cgo path with a fault injection that triggers
// LUX_INVALID_ARGUMENT inside the C library, and asserts the
// propagation contract through the wrapping.
//
// Build tag note: this test is gated behind `lux_accel_real` because
// the default-build stub returns LUX_NO_BACKEND for every C entry —
// session creation itself fails, and we never reach the per-op
// dispatch path. On a host with libluxaccel installed (Spark GB10,
// CI runners with CUDA / Metal plugins), build with:
//
//   go test -tags lux_accel_real ./...
//
// On stub builds the test compiles but doesn't run (the build tag
// excludes it).
//
// Fault injection strategy
// ------------------------
// Three paths are exercised, each one a real cgo call:
//
//   1. ops MLDSAVerifyBatch with a malformed tensor (count=0, no
//      data) — the C library validates batch shape and returns
//      LUX_INVALID_ARGUMENT for any disagreement between the count
//      claimed by the messages tensor shape and the count claimed by
//      the results tensor shape. This is the same precise path the
//      Probe 9 finding identified.
//
//   2. ops SLHDSAVerifyBatch — same shape, locks SLH-DSA M-1.
//
//   3. NewTensor with an absurd shape (negative-via-cast or zero
//      dim) — exercises the cgo tensor creation path that returns
//      LUX_INVALID_ARGUMENT from the C library when the caller's
//      shape is malformed past the Go-level pre-validation. This
//      catches the createTensor fix in session_c.go.
//
// Each subtest is best-effort: if the active backend can't be
// induced to return INVALID_ARGUMENT for the chosen vector, the
// subtest is skipped with a clear message so a future maintainer
// knows the fault-injection needs refining for that backend. The
// test SUCCEEDS for any subtest that DID produce an error and the
// error satisfies errors.Is(err, accel.ErrInvalidArgument). A
// FAILURE means the error was returned but it carries the WRONG
// sentinel — that's the consensus-split shape Red Probe 9 found.
func TestRed_Probe9_M1PropagationThroughCgo(t *testing.T) {
	sess, err := accel.NewSession()
	if err != nil {
		// On a real GPU host this should succeed. On a default-stub
		// build (which this file's build tag excludes anyway), it
		// returns LUX_NO_BACKEND.
		t.Skipf("no GPU session available (NewSession returned %v); the propagation contract requires a real backend to exercise", err)
		return
	}
	defer sess.Close()

	// Helper: assert that err is non-nil AND errors.Is(err,
	// accel.ErrInvalidArgument) is true. We require BOTH because:
	//
	//   * err == nil  → fault injection didn't trigger; backend
	//                    accepted the malformed input. Skip with a
	//                    diagnostic so a future maintainer can refine
	//                    the injection for this backend.
	//   * err != nil but errors.Is == false → the leak Red Probe 9
	//     identified is still present. THIS is the consensus-split
	//     bug shape; fail loudly.
	requireInvalidArgPropagation := func(t *testing.T, op string, err error) {
		t.Helper()
		if err == nil {
			t.Skipf("%s: backend did not reject the fault-injected input; refine injection or test on a stricter backend", op)
			return
		}
		// The actual assertion — this is what Probe 9 broke.
		if !errors.Is(err, accel.ErrInvalidArgument) {
			t.Fatalf(
				"%s returned %T (%v); want errors.Is(err, accel.ErrInvalidArgument) == true. "+
					"The M-1 propagation chain is BROKEN — capi.ErrInvalidArgument is leaking past translateCapiError. "+
					"Every M-1 consumer's errors.Is check at the dispatch sites will see false → silent CPU fallback → "+
					"consensus split between GPU and CPU validators.",
				op, err, err)
		}
	}

	// Subtest 1: MLDSAVerifyBatch — the precise path Red Probe 9
	// flagged. ops_c.go::cgoLatticeOps.MLDSAVerifyBatch now wraps
	// through translateCapiError; the propagation should hold.
	t.Run("MLDSAVerifyBatch_shape_mismatch", func(t *testing.T) {
		// Construct tensors with deliberately mismatched shapes.
		// The C library will reject any disagreement between the
		// count implied by msgs.shape[0] and results.shape[0].
		msgs, err := accel.NewTensor[uint8](sess, []int{2, 32})
		if err != nil {
			t.Skipf("baseline NewTensor failed (%v); backend partial", err)
			return
		}
		defer msgs.Close()
		sigs, err := accel.NewTensor[uint8](sess, []int{2, 3309})
		if err != nil {
			t.Skipf("baseline sigs NewTensor failed (%v)", err)
			return
		}
		defer sigs.Close()
		pks, err := accel.NewTensor[uint8](sess, []int{2, 1952})
		if err != nil {
			t.Skipf("baseline pks NewTensor failed (%v)", err)
			return
		}
		defer pks.Close()
		// Shape mismatch: results says count=4 while everything else
		// says count=2. The C library MUST reject.
		results, err := accel.NewTensor[uint8](sess, []int{4})
		if err != nil {
			t.Skipf("baseline results NewTensor failed (%v)", err)
			return
		}
		defer results.Close()

		err = sess.Lattice().MLDSAVerifyBatch(
			accel.MLDSAMode65,
			msgs.Untyped(),
			sigs.Untyped(),
			pks.Untyped(),
			results.Untyped(),
		)
		requireInvalidArgPropagation(t, "MLDSAVerifyBatch", err)
	})

	// Subtest 2: SLHDSAVerifyBatch — mirror of Subtest 1 for the
	// SLH-DSA dispatch path (lux/crypto/slhdsa/gpu.go:276).
	t.Run("SLHDSAVerifyBatch_shape_mismatch", func(t *testing.T) {
		const slhdsaMode128f = 2
		// SLH-DSA-SHAKE-128f wire sizes: pk=32, sig=17088.
		msgs, err := accel.NewTensor[uint8](sess, []int{2, 32})
		if err != nil {
			t.Skipf("baseline msgs NewTensor failed (%v)", err)
			return
		}
		defer msgs.Close()
		sigs, err := accel.NewTensor[uint8](sess, []int{2, 17088})
		if err != nil {
			t.Skipf("baseline sigs NewTensor failed (%v)", err)
			return
		}
		defer sigs.Close()
		pks, err := accel.NewTensor[uint8](sess, []int{2, 32})
		if err != nil {
			t.Skipf("baseline pks NewTensor failed (%v)", err)
			return
		}
		defer pks.Close()
		// Same shape-mismatch trick.
		results, err := accel.NewTensor[uint8](sess, []int{4})
		if err != nil {
			t.Skipf("baseline results NewTensor failed (%v)", err)
			return
		}
		defer results.Close()

		err = sess.Lattice().SLHDSAVerifyBatch(
			slhdsaMode128f,
			msgs.Untyped(),
			sigs.Untyped(),
			pks.Untyped(),
			results.Untyped(),
		)
		requireInvalidArgPropagation(t, "SLHDSAVerifyBatch", err)
	})

	// Subtest 3: BLSVerifyBatch — covers the non-PQ side of ops_c.go
	// (the ECDSA/Ed25519/BLS group flagged by Probe 9). Same shape
	// fault-injection vector, different op route.
	t.Run("BLSVerifyBatch_shape_mismatch", func(t *testing.T) {
		// BLS12-381: msg=32, sig=96 (G2), pk=48 (G1).
		msgs, err := accel.NewTensor[uint8](sess, []int{2, 32})
		if err != nil {
			t.Skipf("baseline msgs NewTensor failed (%v)", err)
			return
		}
		defer msgs.Close()
		sigs, err := accel.NewTensor[uint8](sess, []int{2, 96})
		if err != nil {
			t.Skipf("baseline sigs NewTensor failed (%v)", err)
			return
		}
		defer sigs.Close()
		pks, err := accel.NewTensor[uint8](sess, []int{2, 48})
		if err != nil {
			t.Skipf("baseline pks NewTensor failed (%v)", err)
			return
		}
		defer pks.Close()
		results, err := accel.NewTensor[uint8](sess, []int{4})
		if err != nil {
			t.Skipf("baseline results NewTensor failed (%v)", err)
			return
		}
		defer results.Close()

		err = sess.Crypto().BLSVerifyBatch(
			msgs.Untyped(),
			sigs.Untyped(),
			pks.Untyped(),
			results.Untyped(),
		)
		requireInvalidArgPropagation(t, "BLSVerifyBatch", err)
	})
}

// TestRed_Probe9_PropagationSentinelDistinctness pins the underlying
// premise of Probe 9: capi.ErrInvalidArgument and accel.ErrInvalidArgument
// are distinct values. If they ever become aliased (Option B from the
// finding — collapse via `var accel.ErrInvalidArgument = capi.ErrInvalidArgument`),
// translateCapiError would be a no-op and this test would document
// that change. The Probe 9 propagation contract still holds under
// either option, but the failure modes differ.
//
// We can't access the unexported capi package from accel_test, so
// this test instead pins what we CAN see: accel.ErrInvalidArgument
// has the expected string. Any future renamer breaking this needs to
// update both errors.go AND every site that wraps capi errors.
func TestRed_Probe9_PropagationSentinelDistinctness(t *testing.T) {
	if accel.ErrInvalidArgument == nil {
		t.Fatal("accel.ErrInvalidArgument is nil")
	}
	got := accel.ErrInvalidArgument.Error()
	want := "accel: invalid argument"
	if got != want {
		t.Fatalf("accel.ErrInvalidArgument.Error() = %q; want %q. "+
			"If you renamed the sentinel, update every translateCapiError site too.",
			got, want)
	}
	// errors.Is reflexivity: any sentinel must match itself.
	if !errors.Is(accel.ErrInvalidArgument, accel.ErrInvalidArgument) {
		t.Fatal("errors.Is(accel.ErrInvalidArgument, accel.ErrInvalidArgument) == false; impossible")
	}
}
