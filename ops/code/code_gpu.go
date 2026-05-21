// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build accel && cgo

// GPU dispatch for code-based crypto. The current implementation
// delegates to the CPU kernels — see the "Why no Metal kernels here"
// commentary in this file. The accel build tag wires this file in
// place of the stub so that the surface is available when other
// accel ops are GPU-enabled, but the actual compute still runs in
// the canonical CPU kernels.
//
// Why no Metal kernels here
// -------------------------
// HQC's hot paths are:
//   * GF(2)^N Karatsuba polynomial multiplication. Inner loop is
//     bit-sliced 64-bit base case (PCLMUL-shaped). On Apple M1 Max
//     the NEON CARRY-LESS multiply (PMULL) gives the CPU a per-cycle
//     throughput Metal can't match for the SMALL vectors HQC uses
//     (277-901 uint64 = 2.2-7.2 KB per polynomial). The kernel
//     launch latency dominates: a Metal MTLCommandBuffer dispatch
//     costs ~5-10 µs while a single CPU polymul costs ~30 µs for
//     HQC-128 and ~250 µs for HQC-256. Below ~32 polynomials per
//     batch the CPU wins outright; above that the bottleneck shifts
//     to host-device transfer of 2-7 KB per slot, where unified
//     memory on M-series chips eliminates the win.
//   * Reed-Solomon decode (Berlekamp-Massey + additive FFT). The
//     control flow is irregular (data-dependent branches in the
//     ELP iteration even though the implementation is constant-time
//     against secret material) and PARAM_DELTA <= 29 means the FFT
//     is over a 32-point domain. Both factors keep this well within
//     CPU territory.
//
// Where GPU dispatch DOES pay off
// ------------------------------
// Batch parallelism across INDEPENDENT operations. The CPU path with
// OpenMP enabled already exploits this — N threads, N independent
// HQC ops, near-linear scaling up to physical core count. The Metal
// path would beat OpenMP only when N exceeds CPU core count by
// roughly the GPU SIMD width (~32 on M1 Max). That regime exists
// (e.g. validator-set BLS-aggregation-replacement workloads with
// thousands of independent encapsulations) but is beyond the scope
// of the initial wiring.
//
// The dispatch stays on the CPU kernels through the GPU session so
// that:
//   1. The accel build tag does not change correctness.
//   2. A future Metal/CUDA backend can drop in by overriding the
//      lux_hqc_* C symbols (the plugin ABI is already in place via
//      the static library that links into libluxgpu).

package code

import (
	"github.com/luxfi/accel"
)

// hqcKeypairGPU dispatches the keypair batch via the GPU session. For
// HQC the session merely owns the lifecycle (so callers can pool a
// long-lived accel.Session for batched workloads); the compute runs
// on the CPU kernels until a Metal/CUDA backend ships the override.
func hqcKeypairGPU(sess *accel.Session, mode Mode, pks, sks, seeds []byte, count int) error {
	_ = sess
	return hqcKeypairCPU(mode, pks, sks, seeds, count)
}

func hqcEncapsGPU(sess *accel.Session, mode Mode, cts, sss, pks, seeds []byte, count int) error {
	_ = sess
	return hqcEncapsCPU(mode, cts, sss, pks, seeds, count)
}

func hqcDecapsGPU(sess *accel.Session, mode Mode, sss, cts, sks []byte, count int) error {
	_ = sess
	return hqcDecapsCPU(mode, sss, cts, sks, count)
}

func gf2PolymulGPU(sess *accel.Session, mode Mode, c, a, b []uint64, count int) error {
	_ = sess
	return gf2PolymulCPU(mode, c, a, b, count)
}

func rsDecodeGPU(sess *accel.Session, mode Mode, msgs, cdws []byte, count int) error {
	_ = sess
	return rsDecodeCPU(mode, msgs, cdws, count)
}
