import Link from "next/link"

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-4">
      <div className="container flex flex-col items-center gap-12 py-24 sm:gap-16 sm:py-32">
        <div className="flex flex-col items-center gap-4 text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
            Lux Accel
          </h1>
          <p className="max-w-2xl text-lg text-muted-foreground">
            GPU-accelerated cryptography, ZK proofs, FHE, and ML operations for Lux blockchain
          </p>
        </div>
        <div className="flex gap-4">
          <Link
            href="/docs"
            className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground shadow transition-colors hover:bg-primary/90"
          >
            Get Started
          </Link>
          <Link
            href="/docs/api"
            className="inline-flex h-10 items-center justify-center rounded-md border border-input bg-background px-8 text-sm font-medium shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            API Reference
          </Link>
        </div>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 max-w-4xl">
          <div className="rounded-lg border bg-card p-6">
            <h3 className="font-semibold mb-2">Multi-Backend</h3>
            <p className="text-sm text-muted-foreground">
              Metal, CUDA, and WebGPU backends with automatic detection
            </p>
          </div>
          <div className="rounded-lg border bg-card p-6">
            <h3 className="font-semibold mb-2">Post-Quantum Crypto</h3>
            <p className="text-sm text-muted-foreground">
              Kyber (ML-KEM) and Dilithium (ML-DSA) NIST standards
            </p>
          </div>
          <div className="rounded-lg border bg-card p-6">
            <h3 className="font-semibold mb-2">ZK Acceleration</h3>
            <p className="text-sm text-muted-foreground">
              NTT, MSM, and polynomial operations for ZK proofs
            </p>
          </div>
          <div className="rounded-lg border bg-card p-6">
            <h3 className="font-semibold mb-2">FHE Operations</h3>
            <p className="text-sm text-muted-foreground">
              BFV and CKKS schemes for homomorphic encryption
            </p>
          </div>
          <div className="rounded-lg border bg-card p-6">
            <h3 className="font-semibold mb-2">ML Kernels</h3>
            <p className="text-sm text-muted-foreground">
              MatMul, attention, convolutions, and activations
            </p>
          </div>
          <div className="rounded-lg border bg-card p-6">
            <h3 className="font-semibold mb-2">Node Integration</h3>
            <p className="text-sm text-muted-foreground">
              Direct integration with Lux node and EVM precompiles
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
