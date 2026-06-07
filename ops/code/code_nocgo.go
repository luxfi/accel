// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo

// Pure-Go stub for the CPU path, used only in no-cgo builds
// (CGO_ENABLED=0). The default build (cgo) links the native HQC
// reference via code_cpu.go; this stub exists solely so the package
// still compiles when cgo is disabled, returning a clean error rather
// than dragging the luxcpp static-lib dependency into a pure-Go build.

package code

// All entry points return ErrNativeHQCUnavailable (defined in code.go):
// native HQC is not linked in a no-cgo build. Build with cgo (the
// default) to link libluxgpu_hqc.

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
