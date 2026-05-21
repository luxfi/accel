// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

// CPU reference implementation for code-based crypto. Links directly
// against the canonical luxgpu_hqc static library (built by
// luxfi/mlx). This is always-available (gated only on cgo) and is the
// byte-equal oracle for every backend, including the GPU dispatch.
//
// The wire format and seed layout exposed by code.go matches the C
// API exposed in luxfi/mlx/include/lux/gpu/hqc.h. Slot indexing and
// buffer sizes are validated at the Go boundary so the C entry points
// can trust their inputs.
//
// Build expects:
//   * libluxgpu_hqc.a built at ${LUX_MLX_BUILD_DIR}/libluxgpu_hqc.a
//     (default: ../mlx/build/libluxgpu_hqc.a relative to the accel repo)
//   * lux/gpu/hqc.h header reachable via the -I path
//
// The defaults assume the standard luxfi sibling-repo layout; override
// via the LUX_MLX_BUILD_DIR and LUX_MLX_INCLUDE_DIR env vars at
// `go build` time using CGO_CFLAGS / CGO_LDFLAGS if the layout differs.

package code

/*
#cgo CFLAGS: -I${SRCDIR}/../../../mlx/include
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../mlx/build -lluxgpu_hqc -lc++ -framework Security
#cgo !darwin LDFLAGS: -L${SRCDIR}/../../../mlx/build -lluxgpu_hqc -lstdc++ -lm

#include <stdint.h>
#include <stddef.h>
#include "lux/gpu/hqc.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// translateError maps LuxHQCError -> Go error.
func translateError(rc C.LuxHQCError) error {
	switch rc {
	case C.LUX_HQC_OK:
		return nil
	case C.LUX_HQC_ERROR_INVALID_ARG:
		return ErrInvalidInput
	case C.LUX_HQC_ERROR_INVALID_MODE:
		return ErrInvalidMode
	case C.LUX_HQC_ERROR_INVALID_LENGTH:
		return ErrBufferSizeInvalid
	case C.LUX_HQC_ERROR_RNG_FAILED:
		return ErrSeedExhausted
	default:
		return fmt.Errorf("code: HQC error code %d", int(rc))
	}
}

// cgoMode maps Mode -> LuxHQCMode (same bit pattern, kept explicit so
// the Go enum can change independent of the C enum if needed).
func cgoMode(mode Mode) C.LuxHQCMode {
	switch mode {
	case HQC128:
		return C.LUX_HQC_128
	case HQC192:
		return C.LUX_HQC_192
	case HQC256:
		return C.LUX_HQC_256
	default:
		return C.LuxHQCMode(255) // intentionally invalid; C side rejects
	}
}

func hqcKeypairCPU(mode Mode, pks, sks, seeds []byte, count int) error {
	rc := C.lux_hqc_keypair_batch(
		cgoMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&pks[0])),
		(*C.uint8_t)(unsafe.Pointer(&sks[0])),
		(*C.uint8_t)(unsafe.Pointer(&seeds[0])),
		C.size_t(count),
	)
	return translateError(rc)
}

func hqcEncapsCPU(mode Mode, cts, sss, pks, seeds []byte, count int) error {
	rc := C.lux_hqc_encaps_batch(
		cgoMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&cts[0])),
		(*C.uint8_t)(unsafe.Pointer(&sss[0])),
		(*C.uint8_t)(unsafe.Pointer(&pks[0])),
		(*C.uint8_t)(unsafe.Pointer(&seeds[0])),
		C.size_t(count),
	)
	return translateError(rc)
}

func hqcDecapsCPU(mode Mode, sss, cts, sks []byte, count int) error {
	rc := C.lux_hqc_decaps_batch(
		cgoMode(mode),
		(*C.uint8_t)(unsafe.Pointer(&sss[0])),
		(*C.uint8_t)(unsafe.Pointer(&cts[0])),
		(*C.uint8_t)(unsafe.Pointer(&sks[0])),
		C.size_t(count),
	)
	return translateError(rc)
}

func gf2PolymulCPU(mode Mode, c, a, b []uint64, count int) error {
	vecN, err := vecNSize64(mode)
	if err != nil {
		return err
	}
	cMode := cgoMode(mode)
	// Loop in Go so a per-slot kernel failure can short-circuit the
	// batch. The C single-op kernel is constant-time and stateless,
	// so this is no slower than a C-side loop (the only overhead is
	// a per-slot cgo crossing — measured at ~150ns on Apple M1 Max).
	for i := 0; i < count; i++ {
		ai := a[i*vecN : (i+1)*vecN]
		bi := b[i*vecN : (i+1)*vecN]
		ci := c[i*vecN : (i+1)*vecN]
		rc := C.lux_hqc_gf2_polymul(
			cMode,
			(*C.uint64_t)(unsafe.Pointer(&ci[0])),
			(*C.uint64_t)(unsafe.Pointer(&ai[0])),
			(*C.uint64_t)(unsafe.Pointer(&bi[0])),
		)
		if rc != C.LUX_HQC_OK {
			return translateError(rc)
		}
	}
	return nil
}

func rsDecodeCPU(mode Mode, msgs, cdws []byte, count int) error {
	n1, k, err := rsDims(mode)
	if err != nil {
		return err
	}
	cMode := cgoMode(mode)
	for i := 0; i < count; i++ {
		msgI := msgs[i*k : (i+1)*k]
		cdwI := cdws[i*n1 : (i+1)*n1]
		rc := C.lux_hqc_reed_solomon_decode(
			cMode,
			(*C.uint8_t)(unsafe.Pointer(&msgI[0])),
			(*C.uint8_t)(unsafe.Pointer(&cdwI[0])),
		)
		if rc != C.LUX_HQC_OK {
			return translateError(rc)
		}
	}
	return nil
}
