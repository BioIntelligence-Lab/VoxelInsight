import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Copy, Check, ChevronsUpDown } from "lucide-react";

export default function IdcCodeView() {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const code = props?.code || "";
  const title = props?.title || "Generated code";

  const handleCopy = async () => {
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch (err) {
      console.error("Copy failed", err);
    }
  };

  return (
    <div className="relative w-full max-w-3xl rounded-xl border border-border bg-gradient-to-br from-card to-background/60 text-card-foreground shadow-sm">
      <div className="flex items-center justify-between px-3 py-2 gap-3">
        <div className="min-w-0 text-sm font-semibold tracking-tight truncate">{title}</div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2"
            onClick={handleCopy}
            aria-label="Copy code"
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2"
            onClick={() => setExpanded((prev) => !prev)}
            aria-label="Toggle code view"
          >
            <ChevronsUpDown className={`h-4 w-4 transition ${expanded ? "rotate-180" : ""}`} />
          </Button>
        </div>
      </div>
      <div className="border-t border-border">
        <ScrollArea className={expanded ? "h-[420px]" : "h-24"}>
          <pre className="px-3 py-2 text-[13px] leading-5 whitespace-pre overflow-x-auto font-mono text-muted-foreground">
            {code || "No code provided."}
          </pre>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
        {!expanded ? (
          <div className="pointer-events-none absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-card" />
        ) : null}
      </div>
    </div>
  );
}
