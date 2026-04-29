// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build !cgo || !lux_crypto_native
// +build !cgo !lux_crypto_native

package crypto

// msmCPU is the pure-Go path. It returns ErrUnsupported so callers can fall
// back to their own MSM implementation (e.g. gnark-crypto MultiExp inside
// the EIP-2537 precompile). Build with `-tags=lux_crypto_native` and a CGO
// toolchain to route through luxcpp/crypto.
func msmCPU(curve Curve, scalars, points [][]byte) ([]byte, error) {
	return nil, ErrUnsupported
}
