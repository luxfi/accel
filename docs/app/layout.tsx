import "./global.css"
import { RootProvider } from "fumadocs-ui/provider"
import { Inter } from "next/font/google"
import type { ReactNode } from "react"

const inter = Inter({
  subsets: ["latin"],
})

export const metadata = {
  title: {
    default: "Lux Accel Documentation",
    template: "%s | Lux Accel",
  },
  description: "GPU-accelerated cryptography and ML operations for Lux blockchain",
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body>
        <RootProvider>
          {children}
        </RootProvider>
      </body>
    </html>
  )
}
