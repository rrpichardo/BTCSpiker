import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

function timeLabel(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString();
}

// predictions arrives newest-first from the API; the chart needs
// chronological (oldest -> newest, left -> right) order.
export default function ScoreChart({ predictions }) {
  const chartData = [...predictions].reverse().map((p) => {
    const ts = p.feature_ts || p.api_ts;
    return {
      ts,
      label: timeLabel(ts),
      score: p.score,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="label"
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          minTickGap={40}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          width={40}
        />
        <Tooltip
          contentStyle={{
            background: "var(--bg-panel)",
            border: "1px solid var(--border)",
            fontSize: 12,
          }}
          labelFormatter={(label) => `time: ${label}`}
          formatter={(value) => [Number(value).toFixed(3), "score"]}
        />
        <Line
          type="monotone"
          dataKey="score"
          stroke="var(--accent)"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
