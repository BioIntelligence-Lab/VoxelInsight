export default function VoxelProgress() {
  const rawPct = Number(props?.pct ?? 0);
  const pct = Number.isFinite(rawPct) ? Math.max(0, Math.min(100, rawPct)) : 0;
  const label = props?.label ?? "Starting";
  const title = props?.title ?? "Workflow progress";
  const status = props?.status ?? "running";
  const rawTotal = Number(props?.total ?? 0);
  const rawCompleted = Number(props?.completed ?? 0);
  const rawFailed = Number(props?.failed ?? 0);
  const total = Number.isFinite(rawTotal) ? Math.max(0, rawTotal) : 0;
  const completed = Number.isFinite(rawCompleted) ? Math.max(0, rawCompleted) : 0;
  const failed = Number.isFinite(rawFailed) ? Math.max(0, rawFailed) : 0;
  const rawCurrentPct = Number(props?.current_pct);
  const currentPct = Number.isFinite(rawCurrentPct)
    ? Math.max(0, Math.min(100, rawCurrentPct))
    : null;
  const hasCounts = total > 0;
  const isError = status === "error" || status === "failed";
  const isPartial = status === "partial" || failed > 0;
  const barColor = isError
    ? "bg-red-500"
    : isPartial
      ? "bg-amber-500"
      : status === "completed"
        ? "bg-green-500"
        : "bg-blue-500";

  return (
    <div
      className="w-full max-w-xl p-3 rounded-lg border"
      role="status"
      aria-live="polite"
      aria-label={`${title}: ${pct}% ${label}`}
    >
      <div className="flex items-center justify-between gap-3 mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="text-xs tabular-nums">{pct}%</div>
      </div>
      <div
        className="h-2 w-full bg-gray-200 rounded overflow-hidden"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
      >
        <div
          className={`h-2 rounded transition-[width] duration-300 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-2 text-sm">{label}</div>
      {hasCounts && (
        <div className="mt-1 text-xs text-gray-500 tabular-nums">
          {completed}/{total} completed
          {failed > 0 ? ` · ${failed} failed` : ""}
        </div>
      )}
      {props?.current_item && (
        <div className="mt-1 text-xs text-gray-500 truncate" title={props.current_item}>
          Current: {props.current_item}
          {currentPct !== null ? ` · ${currentPct}%` : ""}
        </div>
      )}
      {props?.detail && (
        <div className="mt-1 text-xs text-gray-500">{props.detail}</div>
      )}
    </div>
  );
}
