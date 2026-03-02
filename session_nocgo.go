//go:build \!cgo

package accel

import "errors"

var errNoCGo = errors.New("accel: CGo not available, GPU acceleration disabled")

func initLibrary() error              { return nil }
func shutdown()                       {}
func version() string                 { return "none (no cgo)" }
func lastError() string               { return "" }
func backendCount() int               { return 0 }
func deviceCountForBackend(BackendType) int { return 0 }
func availableBackends() []BackendType      { return nil }
func allDevices() []DeviceInfo              { return nil }
func newSession(opts ...SessionOption) (*Session, error) {
	return nil, errNoCGo
}
func newSessionWithBackend(backend BackendType, opts ...SessionOption) (*Session, error) {
	return nil, errNoCGo
}
func newSessionWithDevice(backend BackendType, deviceIndex int, opts ...SessionOption) (*Session, error) {
	return nil, errNoCGo
}

