// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo

// CPU reference implementation for code-based crypto. Links directly
// against the canonical luxgpu_hqc static library (built by luxfi/mlx,
// aka luxcpp/gpu). This is always-available (gated only on cgo) and is
// the byte-equal oracle for every backend, including the GPU dispatch.
//
// The wire format and seed layout exposed by code.go matches the C
// API exposed in luxfi/mlx/include/lux/gpu/hqc.h. Slot indexing and
// buffer sizes are validated at the Go boundary so the C entry points
// can trust their inputs.
//
// Path discovery (backend-agnostic, no env var required)
// -----------------------------------------------------
// The #cgo CFLAGS / LDFLAGS lines below enumerate every standard
// install prefix where libluxgpu_hqc.a + lux/gpu/hqc.h might live.
// The C compiler silently skips -I / -L paths that don't exist, so
// listing them all is harmless — only the prefix that actually has
// the artefacts will contribute to the resolved build.
//
// Fallback chain (in compiler search order):
//   1. CGO_CFLAGS / CGO_LDFLAGS — caller override (env var)
//      e.g. LUX_GPU_PREFIX=/opt/custom pkg-config-style flags
//   2. /usr/local/{include,lib}             — POSIX system install
//   3. /opt/homebrew/{include,lib}          — Homebrew on Apple Silicon
//   4. /opt/homebrew/opt/lux-gpu/{...}      — Homebrew keg-only install
//   5. /usr/local/opt/lux-gpu/{...}         — Homebrew on Intel Mac
//   6. /opt/lux/{include,lib}               — Lux canonical prefix
//   7. ${SRCDIR}/../../../mlx/{include,build} — in-tree dev fallback
//      (works only when the consuming module is in a workspace next
//      to luxfi/mlx; in the Go module cache this resolves to a path
//      that doesn't exist and is silently skipped)
//
// To override discovery at build time without editing this file:
//
//   CGO_CFLAGS="-I/my/install/include" \
//   CGO_LDFLAGS="-L/my/install/lib" \
//   go build ./...
//
// Go cgo concatenates env-var flags AFTER the #cgo directives, so the
// override paths take precedence in linker order.
//
// For programmatic introspection (which path resolved?) see
// accel.Provenance().GPUPaths() in the parent package.

package code

/*
#cgo CFLAGS: -I/usr/local/include
#cgo CFLAGS: -I/opt/homebrew/include
#cgo CFLAGS: -I/opt/homebrew/opt/lux-gpu/include
#cgo CFLAGS: -I/usr/local/opt/lux-gpu/include
#cgo CFLAGS: -I/opt/lux/include
#cgo CFLAGS: -I${SRCDIR}/../../../mlx/include

#cgo darwin LDFLAGS: -L/usr/local/lib
#cgo darwin LDFLAGS: -L/opt/homebrew/lib
#cgo darwin LDFLAGS: -L/opt/homebrew/opt/lux-gpu/lib
#cgo darwin LDFLAGS: -L/usr/local/opt/lux-gpu/lib
#cgo darwin LDFLAGS: -L/opt/lux/lib
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../mlx/build
#cgo darwin LDFLAGS: -lluxgpu_hqc -lc++ -framework Security
// macOS ld emits "search path '<X>' not found" warnings for any -L
// path that doesn't exist. The fallback chain above probes every
// standard prefix, so those warnings are expected and informational;
// they do NOT indicate a build failure. Linux's GNU ld silently
// skips missing -L paths so this is darwin-only noise.

#cgo !darwin LDFLAGS: -L/usr/local/lib
#cgo !darwin LDFLAGS: -L/opt/lux/lib
#cgo !darwin LDFLAGS: -L${SRCDIR}/../../../mlx/build
#cgo !darwin LDFLAGS: -lluxgpu_hqc -lstdc++ -lm

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
