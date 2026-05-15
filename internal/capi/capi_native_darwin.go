//go:build cgo && accel_native && darwin

// Package capi — opt-in native-library link directives.
//
// Default builds resolve every C symbol against the weak stubs in
// stub.go, which keeps the binary buildable on hosts where libluxaccel
// is not installed. Building with `-tags=accel_native` switches the
// link to the real libluxaccel + libluxgpu + libluxcrypto family, at
// which point the weak stubs are overridden and accel.Available()
// reflects the real backend count (Metal on Apple Silicon, CUDA on
// Linux/NVIDIA).
//
// Required host install:
//   /opt/homebrew/lib/libluxaccel.dylib    (or /usr/local/lib on Intel)
//   /opt/homebrew/lib/libluxgpu.dylib
//   /opt/homebrew/lib/libluxcrypto.dylib
// Plus a Metal plugin at one of:
//   /opt/homebrew/lib/lux/plugins/lux_metal.plugin
//   ~/.lux/plugins/lux_metal.plugin
//   $LUX_PLUGIN_PATH/lux_metal.plugin

package capi

/*
#cgo darwin LDFLAGS: -lluxaccel -lluxgpu -lluxcrypto
*/
import "C"
