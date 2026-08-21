// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

//go:build cgo && lux_gpu_pkgconfig && lux_hqc_native

// Optional pkg-config-driven link surface for the luxgpu substrate.
// Enable with `-tags=lux_gpu_pkgconfig` when you want a hard failure
// if `pkg-config luxgpu` doesn't resolve, instead of falling back to
// the multi-prefix probe in code_cpu.go. Mostly useful in CI / build
// systems that have already installed luxgpu via `cmake --install`
// and want to assert the install is wired up.
//
// The default build (no tag) still works without pkg-config — it
// probes every standard install prefix in code_cpu.go's CGO directives.
//
// To target a non-system install prefix from pkg-config:
//
//   PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig \
//     go build -tags=lux_gpu_pkgconfig ./...

package code

/*
#cgo pkg-config: luxgpu
*/
import "C"
