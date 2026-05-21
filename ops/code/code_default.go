// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !accel || !cgo

// Default (no-accel) build wires the GPU entry points to the CPU
// kernels when cgo is available, and to a pure-Go stub otherwise.
//
// The pure-Go stub returns ErrInvalidInput uniformly — code-based
// crypto requires cgo to reach the PQClean reference. This is the
// same posture as crypto/hqc itself: without `-tags=hqc_pqclean` and
// cgo, every operation errors out cleanly. Callers should fall back
// to the crypto/hqc CPU path (which has its own cgo gate) and not
// rely on the accel surface to produce KEM ciphertexts.

package code

import (
	"github.com/luxfi/accel"
)

// hqcKeypairGPU is a no-op stub when accel/cgo is not enabled.
// Production deployments link with cgo so this branch is unreachable
// at runtime; it exists to keep `GOWORK=off go build ./...` working
// on hosts without a C compiler.
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
