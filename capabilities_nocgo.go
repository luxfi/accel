//go:build !cgo

package accel

import (
	"errors"
	"time"
)

// GetCapabilities returns capabilities for a backend (no-op without CGo).
func GetCapabilities(backend BackendType) (*BackendCapabilities, error) {
	return nil, errors.New("accel: CGo not available")
}

// AllCapabilities returns all backend capabilities (empty without CGo).
func AllCapabilities() map[BackendType]*BackendCapabilities {
	return nil
}

// CompareBackends benchmarks operation across backends (no-op without CGo).
func CompareBackends(op OperationType, iterations int) (*BackendComparison, error) {
	return nil, errors.New("accel: CGo not available")
}

func benchmarkOp(session *Session, op OperationType, iterations int) (time.Duration, error) {
	return 0, errors.New("accel: CGo not available")
}

// SelectBestBackend selects the best backend for given operations (no-op without CGo).
func SelectBestBackend(ops []OperationType, preferPerformance bool) (BackendType, error) {
	return BackendAuto, errors.New("accel: no backends available without CGo")
}

// PrintCapabilities prints backend capabilities (no-op without CGo).
func PrintCapabilities() {}
