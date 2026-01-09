import { source } from "@/lib/source"
import type { ReactNode } from "react"
import { DocsLayout } from "fumadocs-ui/layouts/docs"

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout tree={source.pageTree} sidebar={{ defaultOpenLevel: 1 }}>
      {children}
    </DocsLayout>
  )
}
