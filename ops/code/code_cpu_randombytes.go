// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && !hqc_pqclean && lux_hqc_native

// Standalone `randombytes` shim for the accel/ops/code binding.
//
// PQClean's HQC reference (inside libluxgpu_hqc.a) calls `extern int
// randombytes(out, n)` for every entropy byte it needs. When this
// package is consumed in isolation (e.g. by `go test ./ops/code/...`
// or a downstream service that does NOT also link luxfi/crypto/hqc),
// no other TU provides that symbol; this file does.
//
// The build tag `hqc_pqclean` deliberately matches the tag that
// luxfi/crypto/hqc uses to wire in its OWN `randombytes_shim.c`.
// When that tag is set, both crypto/hqc and accel/ops/code are
// linked into the same binary — only crypto/hqc's shim provides
// `randombytes`, and that shim dispatches between the TLS seed
// buffer (for batch ops) and the Go io.Reader (for single-op
// dispatch). Excluding this file in that case prevents the
// duplicate-symbol link error.

package code

/*
#include <stdint.h>
#include <stddef.h>

// Forward declaration of the TLS seed reader exposed by libluxgpu_hqc.
extern int lux_hqc_randombytes(uint8_t* out, size_t n);

// randombytes — strong definition for standalone accel/ops/code use.
// Forwards to the canonical TLS seed reader. The C++ batch kernel
// pre-installs a per-slot seed slice before each inner PQClean
// invocation, so every read here returns deterministic bytes from
// the caller-provided seed buffer.
int randombytes(uint8_t* out, size_t n);
int randombytes(uint8_t* out, size_t n) {
    return lux_hqc_randombytes(out, n);
}
*/
import "C"
