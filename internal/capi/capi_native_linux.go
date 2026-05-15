//go:build cgo && accel_native && linux

// Package capi — opt-in native-library link directives (Linux).
// See capi_native_darwin.go for the policy. Required host install:
//   /usr/local/lib/libluxaccel.so
//   /usr/local/lib/libluxgpu.so
//   /usr/local/lib/libluxcrypto.so
// Plus a CUDA plugin under the configured LUX_PLUGIN_PATH.

package capi

/*
#cgo linux LDFLAGS: -lluxaccel -lluxgpu -lluxcrypto
*/
import "C"
