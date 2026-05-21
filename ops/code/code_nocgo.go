// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo

// Pure-Go (no-cgo) stub for the CPU path. HQC requires the PQClean C
// reference; without cgo we can only fail cleanly. Callers must use
// `GOWORK=on` and a host with a C toolchain (the canonical build
// posture for every Lux Go module that touches PQ crypto).

package code

import "errors"

var errNoCgo = errors.New("code: HQC operations require cgo (build with CGO_ENABLED=1)")

func hqcKeypairCPU(mode Mode, pks, sks, seeds []byte, count int) error {
	_ = mode
	_ = pks
	_ = sks
	_ = seeds
	_ = count
	return errNoCgo
}

func hqcEncapsCPU(mode Mode, cts, sss, pks, seeds []byte, count int) error {
	_ = mode
	_ = cts
	_ = sss
	_ = pks
	_ = seeds
	_ = count
	return errNoCgo
}

func hqcDecapsCPU(mode Mode, sss, cts, sks []byte, count int) error {
	_ = mode
	_ = sss
	_ = cts
	_ = sks
	_ = count
	return errNoCgo
}

func gf2PolymulCPU(mode Mode, c, a, b []uint64, count int) error {
	_ = mode
	_ = c
	_ = a
	_ = b
	_ = count
	return errNoCgo
}

func rsDecodeCPU(mode Mode, msgs, cdws []byte, count int) error {
	_ = mode
	_ = msgs
	_ = cdws
	_ = count
	return errNoCgo
}
