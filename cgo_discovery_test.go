// Copyright (C) 2024-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package accel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPathReport_EnvPrefixWins(t *testing.T) {
	// Build a fake install layout in a tempdir that has BOTH the
	// header and the static library at the right offsets, then point
	// LUX_GPU_PREFIX at it. Discovery MUST resolve to SourceEnv with
	// matching paths.
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "include", "lux", "gpu", "hqc.h"), "// stub")
	mustWriteFile(t, filepath.Join(dir, "lib", staticLibName()), "stub-archive")

	t.Setenv("LUX_GPU_PREFIX", dir)
	t.Setenv("LUX_MLX_PREFIX", "")

	rep := GPUPaths()
	if rep.Source != SourceEnv {
		t.Fatalf("expected SourceEnv (%q), got %q (candidates=%+v)",
			SourceEnv, rep.Source, rep.Candidates)
	}
	if rep.IncludeDir != filepath.Join(dir, "include") {
		t.Errorf("IncludeDir = %q, want %q", rep.IncludeDir, filepath.Join(dir, "include"))
	}
	if rep.LibDir != filepath.Join(dir, "lib") {
		t.Errorf("LibDir = %q, want %q", rep.LibDir, filepath.Join(dir, "lib"))
	}
	if rep.Library == "" {
		t.Errorf("Library is empty; expected the static library path")
	}
}

func TestPathReport_BackCompatEnvPrefix(t *testing.T) {
	// LUX_MLX_PREFIX is the legacy name and must keep working when
	// LUX_GPU_PREFIX is unset.
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "include", "lux", "gpu", "hqc.h"), "// stub")
	mustWriteFile(t, filepath.Join(dir, "lib", staticLibName()), "stub-archive")

	t.Setenv("LUX_GPU_PREFIX", "")
	t.Setenv("LUX_MLX_PREFIX", dir)

	rep := GPUPaths()
	if rep.Source != SourceEnv {
		t.Fatalf("expected SourceEnv from back-compat LUX_MLX_PREFIX, got %q", rep.Source)
	}
}

func TestPathReport_EnvWithBuildDir(t *testing.T) {
	// The env-prefix lookup also probes `prefix/build` as a sibling
	// of `prefix/include` so an unbundled CMake checkout (where
	// build artefacts live in a `build` subdir, not `lib`) works
	// without manually installing.
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "include", "lux", "gpu", "hqc.h"), "// stub")
	mustWriteFile(t, filepath.Join(dir, "build", staticLibName()), "stub-archive")

	t.Setenv("LUX_GPU_PREFIX", dir)
	t.Setenv("LUX_MLX_PREFIX", "")

	rep := GPUPaths()
	if rep.Source != SourceEnv {
		t.Fatalf("expected SourceEnv (build-dir variant), got %q", rep.Source)
	}
	if rep.LibDir != filepath.Join(dir, "build") {
		t.Errorf("LibDir = %q, want %q", rep.LibDir, filepath.Join(dir, "build"))
	}
}

func TestPathReport_MissingWhenNothingInstalled(t *testing.T) {
	// Point env prefixes at a tempdir with NEITHER the header nor
	// the lib. Discovery should fall through every candidate and
	// land on SourceMissing (assuming the host also doesn't have a
	// system install at /usr/local; we can't easily isolate that
	// in a unit test, so we accept either Missing OR a real system
	// install — the assertion is on Candidates always being non-empty).
	dir := t.TempDir() // empty
	t.Setenv("LUX_GPU_PREFIX", dir)
	t.Setenv("LUX_MLX_PREFIX", "")

	rep := GPUPaths()
	if len(rep.Candidates) == 0 {
		t.Fatal("Candidates is empty; discovery should always enumerate the fallback chain")
	}
	// Whatever resolved, it must NOT be the empty env prefix (since
	// no files exist there).
	for _, c := range rep.Candidates {
		if c.Source == SourceEnv && (c.IncludeOK || c.LibOK) {
			t.Errorf("empty env-prefix candidate appears resolved: %+v", c)
		}
	}
}

func TestPathReport_CandidatesInOrder(t *testing.T) {
	// Verify the documented fallback chain order — readers of the
	// PathReport should be able to trust the candidates slice as a
	// faithful enumeration of what was probed.
	t.Setenv("LUX_GPU_PREFIX", "/some/override")
	t.Setenv("LUX_MLX_PREFIX", "")
	rep := GPUPaths()

	wantOrder := []Source{
		SourceEnv,            // 1
		SourceEnv,            // 1b (build-dir variant)
		SourceHomebrewKeg,    // 4
		SourceHomebrewIntel,  // 5
		SourceHomebrewARM,    // 6
		SourceSystem,         // 7
		SourceLuxPrefix,      // 8
		SourceModuleRelative, // 9
	}
	// Optional layers (CgoEnv, pkg-config) may or may not appear
	// depending on the host environment; filter them out before
	// comparing the strict order.
	gotOrder := []Source{}
	for _, c := range rep.Candidates {
		if c.Source == SourceCgoEnv || c.Source == SourcePkgConfig {
			continue
		}
		gotOrder = append(gotOrder, c.Source)
	}
	if len(gotOrder) != len(wantOrder) {
		t.Fatalf("candidate count = %d, want %d (got=%+v)",
			len(gotOrder), len(wantOrder), gotOrder)
	}
	for i := range wantOrder {
		if gotOrder[i] != wantOrder[i] {
			t.Errorf("candidate[%d].Source = %q, want %q",
				i, gotOrder[i], wantOrder[i])
		}
	}
}

func TestPathReport_LogsResolved(t *testing.T) {
	// Reviewer-friendly log line: prints the resolved path + source
	// so PR reviewers can see which fallback the test machine used.
	// This is not an assertion; it's there so `go test -v` documents
	// the discovery outcome.
	rep := GPUPaths()
	t.Logf("Resolved GPU substrate: source=%s include=%s lib=%s library=%s",
		rep.Source, rep.IncludeDir, rep.LibDir, rep.Library)
	t.Logf("Candidates probed (%d):", len(rep.Candidates))
	for _, c := range rep.Candidates {
		t.Logf("  [%s] include=%s libdir=%s incOK=%t libOK=%t",
			c.Source, c.IncludeDir, c.LibDir, c.IncludeOK, c.LibOK)
	}
}

func TestFirstFlagPath(t *testing.T) {
	cases := []struct {
		name   string
		flags  string
		prefix string
		want   string
	}{
		{"single I", "-I/usr/local/include", "-I", "/usr/local/include"},
		{"multiple I", "-I/first -I/second", "-I", "/first"},
		{"single L", "-L/usr/local/lib", "-L", "/usr/local/lib"},
		{"mixed", "-O2 -I/foo -L/bar -lqux", "-L", "/bar"},
		{"missing", "-I/foo", "-L", ""},
		{"empty", "", "-I", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := firstFlagPath(tc.flags, tc.prefix)
			if got != tc.want {
				t.Errorf("firstFlagPath(%q, %q) = %q, want %q",
					tc.flags, tc.prefix, got, tc.want)
			}
		})
	}
}

func TestStaticLibName_GoosAware(t *testing.T) {
	// staticLibName uses runtime.GOOS, not GOOS at test-author time.
	// On POSIX this returns the .a name; on Windows the .lib name.
	// We can only assert one half of that here; the other half is
	// covered by the import-time runtime.GOOS branch.
	got := staticLibName()
	if !strings.HasPrefix(got, "libluxgpu_hqc") && !strings.HasPrefix(got, "luxgpu_hqc") {
		t.Errorf("staticLibName() = %q, expected prefix libluxgpu_hqc or luxgpu_hqc", got)
	}
}

func mustWriteFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
