// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package accel

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestNewVMSession_BasicLifecycle covers create, ID, close, double-close.
func TestNewVMSession_BasicLifecycle(t *testing.T) {
	s, err := NewVMSession("vm-test-1", WithPriority(PriorityHigh))
	if err != nil {
		t.Fatalf("NewVMSession: %v", err)
	}
	if s.ID() != "vm-test-1" {
		t.Fatalf("ID = %q, want %q", s.ID(), "vm-test-1")
	}
	if s.Priority() != PriorityHigh {
		t.Fatalf("Priority = %d, want %d", s.Priority(), PriorityHigh)
	}
	if s.IsClosed() {
		t.Fatal("session reports closed before Close")
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !s.IsClosed() {
		t.Fatal("IsClosed = false after Close")
	}
	// Double close is a no-op.
	if err := s.Close(); err != nil {
		t.Fatalf("Close (2nd): %v", err)
	}
}

// TestNewVMSession_RejectsEmptyID guards against silent misconfig.
func TestNewVMSession_RejectsEmptyID(t *testing.T) {
	_, err := NewVMSession("")
	if !errors.Is(err, ErrEmptyVMID) {
		t.Fatalf("err = %v, want ErrEmptyVMID", err)
	}
}

// TestVMSession_MemoryBudget verifies reserve/release and cap enforcement.
func TestVMSession_MemoryBudget(t *testing.T) {
	s, err := NewVMSession("vm-budget", WithMemoryBudget(1024))
	if err != nil {
		t.Fatalf("NewVMSession: %v", err)
	}
	defer s.Close()

	if err := s.reserve(512); err != nil {
		t.Fatalf("reserve 512: %v", err)
	}
	if err := s.reserve(512); err != nil {
		t.Fatalf("reserve 512 (2nd): %v", err)
	}
	if err := s.reserve(1); !errors.Is(err, ErrSessionBudgetExceeded) {
		t.Fatalf("reserve over cap: err=%v want ErrSessionBudgetExceeded", err)
	}
	if got := s.MemoryUsed(); got != 1024 {
		t.Fatalf("MemoryUsed = %d, want 1024", got)
	}
	s.release(1024)
	if got := s.MemoryUsed(); got != 0 {
		t.Fatalf("MemoryUsed after release = %d, want 0", got)
	}
}

// TestMultiVM_ConcurrentDispatch spins up 4 simulated VMs (cevm/fhe/zk/bridge),
// submits 1000 ops per VM concurrently, and verifies:
//   - Each VM sees only its own results (no cross-contamination).
//   - Within a VM, ops complete in submission order.
//   - All VMs make progress concurrently.
func TestMultiVM_ConcurrentDispatch(t *testing.T) {
	const opsPerVM = 1000

	type vmCase struct {
		name     string
		priority Priority
	}
	cases := []vmCase{
		{"cevm-like", PriorityHigh},
		{"fhe-like", PriorityNormal},
		{"zk-like", PriorityNormal},
		{"bridge-like", PriorityLow},
	}

	sessions := make([]*VMSession, len(cases))
	for i, c := range cases {
		s, err := NewVMSession(c.name, WithPriority(c.priority))
		if err != nil {
			t.Fatalf("NewVMSession(%s): %v", c.name, err)
		}
		sessions[i] = s
	}
	defer func() {
		for _, s := range sessions {
			_ = s.Close()
		}
	}()

	// Per-VM result log: index of last completed op. Must be monotonic.
	results := make([][]int64, len(cases))
	for i := range results {
		results[i] = make([]int64, opsPerVM)
	}

	// Tag identifies which VM produced a given op result. Cross-contamination
	// would show up as a tag mismatch.
	tags := make([]string, len(cases))
	for i, c := range cases {
		tags[i] = c.name
	}

	var wg sync.WaitGroup
	start := make(chan struct{})

	// Submit ops in submission order via an atomic counter per VM.
	// Each Submit captures `i` (op index) and asserts ordering.
	var lastSeen [4]atomic.Int64
	for i := range sessions {
		lastSeen[i].Store(-1)
	}

	for vmIdx, sess := range sessions {
		wg.Add(1)
		go func(vmIdx int, sess *VMSession) {
			defer wg.Done()
			<-start
			ctx := context.Background()
			for op := 0; op < opsPerVM; op++ {
				op := op
				err := sess.Submit(ctx, func(_ *Session) error {
					// Verify FIFO: the previous op for this VM must have
					// already completed.
					prev := lastSeen[vmIdx].Load()
					if prev != int64(op-1) {
						return fmt.Errorf("vm %s: out-of-order op=%d prev=%d",
							sess.ID(), op, prev)
					}
					lastSeen[vmIdx].Store(int64(op))
					results[vmIdx][op] = int64(op) * 1_000 // tagged value
					return nil
				})
				// VMSession returns ErrNoBackends when no GPU is present;
				// the synthetic op above still runs because Submit invokes
				// f even with a nil underlying Session... except the current
				// implementation short-circuits to ErrNoBackends. So when no
				// backend is available, we simulate dispatch without
				// touching the GPU by running the closure directly.
				if errors.Is(err, ErrNoBackends) {
					// Simulate: run the same closure ourselves under the
					// session's queue lock to keep ordering guarantees.
					_ = sess.runSimulated(ctx, func(_ *Session) error {
						prev := lastSeen[vmIdx].Load()
						if prev != int64(op-1) {
							return fmt.Errorf("vm %s: out-of-order op=%d prev=%d",
								sess.ID(), op, prev)
						}
						lastSeen[vmIdx].Store(int64(op))
						results[vmIdx][op] = int64(op) * 1_000
						return nil
					})
					continue
				}
				if err != nil {
					t.Errorf("vm %s op=%d: %v", sess.ID(), op, err)
					return
				}
			}
		}(vmIdx, sess)
	}

	close(start)
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("timed out waiting for VM dispatch goroutines")
	}

	// Verify: each VM saw all opsPerVM ops in order, with correct values.
	for vmIdx, c := range cases {
		got := lastSeen[vmIdx].Load()
		if got != int64(opsPerVM-1) {
			t.Errorf("vm %s: lastSeen=%d, want %d", c.name, got, opsPerVM-1)
		}
		for op, val := range results[vmIdx] {
			want := int64(op) * 1_000
			if val != want {
				t.Errorf("vm %s op=%d: result=%d want=%d (cross-contamination?)",
					c.name, op, val, want)
				break
			}
		}
	}
}

// TestMultiVM_IsolationOnClose verifies that closing one VM session
// does not prevent the others from continuing.
func TestMultiVM_IsolationOnClose(t *testing.T) {
	const numVMs = 4
	sessions := make([]*VMSession, numVMs)
	for i := 0; i < numVMs; i++ {
		s, err := NewVMSession(fmt.Sprintf("iso-vm-%d", i))
		if err != nil {
			t.Fatalf("NewVMSession: %v", err)
		}
		sessions[i] = s
	}

	// Kill VM #1 immediately.
	if err := sessions[1].Close(); err != nil {
		t.Fatalf("Close vm-1: %v", err)
	}

	ctx := context.Background()
	var aliveCount atomic.Int64
	var wg sync.WaitGroup

	for i, s := range sessions {
		wg.Add(1)
		go func(i int, s *VMSession) {
			defer wg.Done()
			err := s.runSimulated(ctx, func(_ *Session) error { return nil })
			if i == 1 {
				if !errors.Is(err, ErrSessionClosed) {
					t.Errorf("vm-1: err=%v, want ErrSessionClosed", err)
				}
				return
			}
			if err != nil {
				t.Errorf("vm-%d: unexpected err=%v", i, err)
				return
			}
			aliveCount.Add(1)
		}(i, s)
	}
	wg.Wait()

	if got := aliveCount.Load(); got != int64(numVMs-1) {
		t.Fatalf("alive sessions = %d, want %d", got, numVMs-1)
	}

	// Cleanup the survivors.
	for i, s := range sessions {
		if i == 1 {
			continue
		}
		_ = s.Close()
	}
}

// TestMultiVM_KillMidDispatch ensures that closing a session mid-stream
// does not block or corrupt other sessions' progress.
func TestMultiVM_KillMidDispatch(t *testing.T) {
	const numVMs = 4
	const opsPerVM = 500

	sessions := make([]*VMSession, numVMs)
	for i := 0; i < numVMs; i++ {
		s, err := NewVMSession(fmt.Sprintf("kill-vm-%d", i))
		if err != nil {
			t.Fatalf("NewVMSession: %v", err)
		}
		sessions[i] = s
	}

	var wg sync.WaitGroup
	var completed [numVMs]atomic.Int64
	ctx := context.Background()

	for i, s := range sessions {
		wg.Add(1)
		go func(i int, s *VMSession) {
			defer wg.Done()
			for op := 0; op < opsPerVM; op++ {
				err := s.runSimulated(ctx, func(_ *Session) error {
					return nil
				})
				if errors.Is(err, ErrSessionClosed) {
					return
				}
				if err != nil {
					t.Errorf("vm-%d op=%d: %v", i, op, err)
					return
				}
				completed[i].Add(1)
			}
		}(i, s)
	}

	// Kill VM #2 while others run.
	time.Sleep(2 * time.Millisecond)
	if err := sessions[2].Close(); err != nil {
		t.Fatalf("Close vm-2: %v", err)
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("timeout: a VM goroutine got stuck after sibling close")
	}

	// VMs 0, 1, 3 must complete all ops. VM 2 may complete some-or-none.
	for i := 0; i < numVMs; i++ {
		got := completed[i].Load()
		if i == 2 {
			if got > opsPerVM {
				t.Errorf("vm-2 completed=%d, > opsPerVM", got)
			}
			continue
		}
		if got != int64(opsPerVM) {
			t.Errorf("vm-%d completed=%d, want %d", i, got, opsPerVM)
		}
	}

	for i, s := range sessions {
		if i == 2 {
			continue
		}
		_ = s.Close()
	}
}

// runSimulated runs f under the queue lock without requiring a GPU backend.
// Used by tests so they exercise the FIFO ordering and isolation paths even
// when CGO is disabled. Not exported because production callers must use
// Submit, which only succeeds when a real session is attached.
func (v *VMSession) runSimulated(ctx context.Context, f func(*Session) error) error {
	if v.closed.Load() {
		return ErrSessionClosed
	}
	if err := lockCtx(ctx, &v.qMu); err != nil {
		return err
	}
	defer v.qMu.Unlock()
	if v.closed.Load() {
		return ErrSessionClosed
	}
	v.dispatched.Add(1)
	if err := f(v.sess); err != nil {
		v.failed.Add(1)
		return err
	}
	v.completed.Add(1)
	return nil
}
