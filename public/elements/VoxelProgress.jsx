export default function VoxelProgress(props) {
  const pct = Math.max(0, Math.min(100, Number(props.pct ?? 0)));
  const label = props.label ?? "Starting";

  return (
    <div className="w-full max-w-xl p-3 rounded-lg border">
      <div className="h-2 w-full bg-gray-200 rounded">
        <div
          className="h-2 rounded bg-blue-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-2 text-sm">
        {pct}% — {label}
      </div>
    </div>
  );
}
