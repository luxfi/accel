// Copyright (C) 2025, Lux Industries, Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package accel

import (
	"errors"
	"testing"
)

func TestNoopBehavior(t *testing.T) {
	// These tests verify the noop pattern works correctly.
	// When CGO is disabled, all functions should return ErrNotSupported.

	t.Run("Available", func(t *testing.T) {
		// Available() should return false when no backends
		if Available() && len(Backends()) == 0 {
			t.Error("Available() should return false when no backends")
		}
	})

	t.Run("MetalAvailable", func(t *testing.T) {
		metal := MetalAvailable()
		// If Metal not in Backends(), should be false
		hasMetalBackend := false
		for _, b := range Backends() {
			if b == BackendMetal {
				hasMetalBackend = true
				break
			}
		}
		if metal != hasMetalBackend {
			t.Errorf("MetalAvailable() = %v, but hasMetalBackend = %v", metal, hasMetalBackend)
		}
	})

	t.Run("DefaultSession", func(t *testing.T) {
		session, err := DefaultSession()
		if Available() {
			if err != nil {
				t.Errorf("DefaultSession() failed when available: %v", err)
			}
			if session == nil {
				t.Error("DefaultSession() returned nil session when available")
			}
		} else {
			if !errors.Is(err, ErrNoBackends) {
				t.Errorf("DefaultSession() error = %v, want ErrNoBackends", err)
			}
			if session != nil {
				t.Error("DefaultSession() should return nil session when unavailable")
			}
		}
	})

	t.Run("BLSBatchVerify", func(t *testing.T) {
		_, err := BLSBatchVerify(nil, nil, nil)
		if !Available() && !errors.Is(err, ErrNotSupported) && !errors.Is(err, ErrBatchSizeMismatch) {
			t.Errorf("BLSBatchVerify() error = %v, want ErrNotSupported or ErrBatchSizeMismatch", err)
		}
	})

	t.Run("SHA256Batch", func(t *testing.T) {
		_, err := SHA256Batch(nil)
		if !Available() && !errors.Is(err, ErrNotSupported) && !errors.Is(err, ErrBatchSizeMismatch) {
			t.Errorf("SHA256Batch() error = %v, want ErrNotSupported or ErrBatchSizeMismatch", err)
		}
	})

	t.Run("KyberKeyGen", func(t *testing.T) {
		_, _, err := KyberKeyGen()
		if !Available() && !errors.Is(err, ErrNotSupported) {
			t.Errorf("KyberKeyGen() error = %v, want ErrNotSupported", err)
		}
	})

	t.Run("DilithiumSign", func(t *testing.T) {
		_, err := DilithiumSign(nil, nil)
		if !Available() && !errors.Is(err, ErrNotSupported) {
			t.Errorf("DilithiumSign() error = %v, want ErrNotSupported", err)
		}
	})

	t.Run("NTTForward", func(t *testing.T) {
		err := NTTForward(nil, nil, 0)
		if !Available() && !errors.Is(err, ErrNotSupported) {
			t.Errorf("NTTForward() error = %v, want ErrNotSupported", err)
		}
	})

	t.Run("MSM", func(t *testing.T) {
		_, err := MSM(nil, nil)
		if !Available() && !errors.Is(err, ErrNotSupported) && !errors.Is(err, ErrBatchSizeMismatch) {
			t.Errorf("MSM() error = %v, want ErrNotSupported or ErrBatchSizeMismatch", err)
		}
	})
}

func TestConstants(t *testing.T) {
	// Verify constants are set correctly
	if BLSBatchVerifyThreshold <= 0 {
		t.Error("BLSBatchVerifyThreshold should be positive")
	}
	if HashBatchThreshold <= 0 {
		t.Error("HashBatchThreshold should be positive")
	}
	if KyberPublicKeySize <= 0 {
		t.Error("KyberPublicKeySize should be positive")
	}
	if DilithiumPublicKeySize <= 0 {
		t.Error("DilithiumPublicKeySize should be positive")
	}
}
