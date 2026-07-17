import { useId, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { buildChartData } from "../chartData.js";

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function timeLabel(timestamp) {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "Unavailable" : timeFormatter.format(date);
}

export default function ScoreChart({ predictions }) {
  const summaryId = useId();
  const chartData = useMemo(() => buildChartData(predictions), [predictions]);
  const summary = useMemo(() => {
    if (chartData.length === 0) return "No valid score points are available.";
    const scores = chartData.map((point) => point.score);
    const latest = chartData[chartData.length - 1];
    return `${chartData.length} score points from ${timeLabel(chartData[0].timestamp)} to ${timeLabel(latest.timestamp)}. Latest ${latest.score.toFixed(3)}, range ${Math.min(...scores).toFixed(3)} to ${Math.max(...scores).toFixed(3)}.`;
  }, [chartData]);

  return (
    <figure
      className="score-chart"
      aria-labelledby="score-chart-heading"
      aria-describedby={summaryId}
    >
      <figcaption className="sr-only" id={summaryId}>{summary}</figcaption>
      <div className="score-chart-visual" role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={timeLabel}
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
                border: "1px solid var(--border-strong)",
                borderRadius: 4,
                fontSize: 12,
              }}
              labelFormatter={(label) => `Time: ${timeLabel(label)}`}
              formatter={(value) => [Number(value).toFixed(3), "Score"]}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="var(--accent)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
