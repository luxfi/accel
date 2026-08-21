// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo || !lux_hqc_native

// Pure-Go stub for the CPU path, used whenever native HQC is NOT linked
// in — i.e. the DEFAULT build (cgo without `-tags=lux_hqc_native`) and
// any no-cgo build. HQC requires the PQClean C reference (or a runtime
// backend); without the native link we fail cleanly rather than drag a
// luxcpp static-lib dependency into every build. Opt in with
// `-tags=lux_hqc_native` (see code_cpu.go) to link libluxgpu_hqc.

package code

// All entry points return ErrNativeHQCUnavailable (defined in code.go):
// native HQC is not linked in this build. Opt in with -tags=lux_hqc_native.

func hqcKeypairCPU(mode Mode, pks, sks, seeds []byte, count int) error {
	_ = mode
	_ = pks
	_ = sks
	_ = seeds
	_ = count
	return ErrNativeHQCUnavailable
}

func hqcEncapsCPU(mode Mode, cts, sss, pks, seeds []byte, count int) error {
	_ = mode
	_ = cts
	_ = sss
	_ = pks
	_ = seeds
	_ = count
	return ErrNativeHQCUnavailable
}

func hqcDecapsCPU(mode Mode, sss, cts, sks []byte, count int) error {
	_ = mode
	_ = sss
	_ = cts
	_ = sks
	_ = count
	return ErrNativeHQCUnavailable
}

func gf2PolymulCPU(mode Mode, c, a, b []uint64, count int) error {
	_ = mode
	_ = c
	_ = a
	_ = b
	_ = count
	return ErrNativeHQCUnavailable
}

func rsDecodeCPU(mode Mode, msgs, cdws []byte, count int) error {
	_ = mode
	_ = msgs
	_ = cdws
	_ = count
	return ErrNativeHQCUnavailable
}
