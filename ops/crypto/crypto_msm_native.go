// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && lux_crypto_native
// +build cgo,lux_crypto_native

package crypto

// msmCPU routes through luxcpp/crypto's gpukit_multi_pippenger_cpu C-ABI
// for all four wired curves (secp256k1, bn254 g1, bls12-381 g1, banderwagon).
// The same signed-digit Bos-Coster Pippenger body backs all of them.
//
// Build with `-tags=lux_crypto_native` to opt in. The C library
// libgpukit_multi_pippenger.a (and its first-party transitive deps) must be
// linked at the standard luxcpp/crypto install prefix.

/*
#cgo CFLAGS: -I${SRCDIR}/../../../../luxcpp/crypto/gpukit/include
#cgo CFLAGS: -I${SRCDIR}/../../../../luxcpp/crypto/include

#cgo darwin LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/gpukit
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/secp256k1
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/bn254
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/banderwagon
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/keccak

#cgo linux LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/gpukit
#cgo linux LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/secp256k1
#cgo linux LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/bn254
#cgo linux LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/banderwagon
#cgo linux LDFLAGS: -L${SRCDIR}/../../../../luxcpp/crypto/build/keccak

#cgo LDFLAGS: -lgpukit_multi_pippenger -lgpukit_cpu
#cgo LDFLAGS: -lsecp256k1_cpu -lbn254_cpu -lbanderwagon_cpu -lkeccak_cpu
#cgo darwin LDFLAGS: -lc++
#cgo linux LDFLAGS: -lstdc++

#include <lux/gpukit/multi_pippenger.h>
#include <lux/gpukit/gpukit.h>
*/
import "C"

import (
	"unsafe"
)

func msmCPU(curve Curve, scalars, points [][]byte) ([]byte, error) {
	n := len(scalars)
	if n == 0 || n != len(points) {
		return nil, ErrInvalidInput
	}
	if curve > CurveBanderwagon {
		return nil, ErrUnsupported
	}
	const scalarSize = 32
	pointSize := curve.pointSize()

	scalarBuf := make([]byte, n*scalarSize)
	pointBuf := make([]byte, n*pointSize)
	for i := 0; i < n; i++ {
		if len(scalars[i]) != scalarSize {
			return nil, ErrInvalidInput
		}
		if len(points[i]) != pointSize {
			return nil, ErrInvalidInput
		}
		copy(scalarBuf[i*scalarSize:], scalars[i])
		copy(pointBuf[i*pointSize:], points[i])
	}

	out := make([]byte, pointSize)
	st := C.gpukit_multi_pippenger_cpu(
		C.uint32_t(curve),
		(*C.uint8_t)(unsafe.Pointer(&scalarBuf[0])),
		(*C.uint8_t)(unsafe.Pointer(&pointBuf[0])),
		C.size_t(n),
		(*C.uint8_t)(unsafe.Pointer(&out[0])),
	)
	if st != C.GPUKIT_OK {
		return nil, ErrVerifyFailed
	}
	return out, nil
}
