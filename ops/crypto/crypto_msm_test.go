// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package crypto

import (
	"errors"
	"testing"
)

func TestMSM_InputValidation(t *testing.T) {
	if _, err := MSM(CurveBLS12_381, nil, nil); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("empty inputs: want ErrInvalidInput, got %v", err)
	}
	if _, err := MSM(CurveBLS12_381,
		[][]byte{make([]byte, 32)},
		[][]byte{make([]byte, 96), make([]byte, 96)},
	); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("len mismatch: want ErrInvalidInput, got %v", err)
	}
}

// TestMSM_PointSize asserts the wire format rule: BLS12-381 G1 is 96 bytes,
// every other supported curve is 64 bytes.
func TestMSM_PointSize(t *testing.T) {
	cases := []struct {
		curve Curve
		want  int
	}{
		{CurveSecp256k1, 64},
		{CurveBN254, 64},
		{CurveBLS12_381, 96},
		{CurveBanderwagon, 64},
	}
	for _, c := range cases {
		if got := c.curve.pointSize(); got != c.want {
			t.Errorf("Curve(%d) pointSize = %d, want %d", c.curve, got, c.want)
		}
	}
}
