import { useId, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { outcomeBands } from "../performanceData.js";
import InfoIcon from "./InfoIcon.jsx";

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function timeLabel(timestamp) {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "Unavailable" : timeFormatter.format(date);
}

// Score-over-time chart for the active grading mode. Score is always
// plotted; actual-spike shading, the alert threshold, and the baseline
// score are independent toggles layered on top.
export default function EvalChart({ chartData, outcomeKey, tau }) {
  const [showBands, setShowBands] = useState(true);
  const [showTau, setShowTau] = useState(true);
  const [showBaseline, setShowBaseline] = useState(true);
  const summaryId = useId();

  const bands = useMemo(
    () => (showBands ? outcomeBands(chartData, outcomeKey) : []),
    [chartData, outcomeKey, showBands],
  );

  const hasTau = showTau && typeof tau === "number" && Number.isFinite(tau);
  const summary = `Score chart with ${chartData.length} graded points.`;

  return (
    <figure className="eval-chart" aria-labelledby="eval-chart-heading" aria-describedby={summaryId}>
      <figcaption className="sr-only" id={summaryId}>
        {summary}
      </figcaption>

      <div className="eval-chart-legend" role="group" aria-label="Chart overlays">
        <label className="eval-toggle">
          <input
            type="checkbox"
            checked={showBands}
            onChange={(event) => setShowBands(event.target.checked)}
          />
          Actual spikes
          <InfoIcon label="Shaded = a real spike actually happened there, measured 60 seconds after the fact." />
        </label>
        <label className="eval-toggle">
          <input
            type="checkbox"
            checked={showTau}
            disabled={typeof tau !== "number"}
            onChange={(event) => setShowTau(event.target.checked)}
          />
          Alert threshold τ
          <InfoIcon label="Scores above this line count as spike alerts. This value ships with the deployed model." />
        </label>
        <label className="eval-toggle">
          <input
            type="checkbox"
            checked={showBaseline}
            onChange={(event) => setShowBaseline(event.target.checked)}
          />
          Baseline
          <InfoIcon label="The naive fixed-threshold rule the ML model has to beat." />
        </label>
      </div>

      <div className="score-chart-visual" role="img" aria-label={summary}>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 8, right: 40, left: 0, bottom: 8 }}>
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
            <YAxis domain={[0, 1]} tick={{ fill: "var(--text-muted)", fontSize: 11 }} width={40} />
            <Tooltip
              contentStyle={{
                background: "var(--bg-panel)",
                border: "1px solid var(--border-strong)",
                borderRadius: 4,
                fontSize: 12,
              }}
              labelFormatter={(label) => `Time: ${timeLabel(label)}`}
              formatter={(value, name) => [Number(value).toFixed(3), name]}
            />
            {bands.map((band, index) => (
              <ReferenceArea
                key={`${band.x1}-${band.x2}-${index}`}
                x1={band.x1}
                x2={band.x2}
                fill="var(--green)"
                fillOpacity={0.16}
                stroke="none"
                ifOverflow="extendDomain"
              />
            ))}
            {hasTau && (
              <ReferenceLine
                y={tau}
                stroke="var(--amber)"
                strokeDasharray="5 4"
                label={{
                  value: `τ ${tau.toFixed(3)}`,
                  position: "right",
                  fill: "var(--amber)",
                  fontSize: 10,
                }}
              />
            )}
            {showBaseline && (
              <Line
                type="stepAfter"
                dataKey="baseline_score"
                name="Baseline"
                stroke="var(--gray)"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            )}
            <Line
              type="monotone"
              dataKey="score"
              name="Score"
              stroke="var(--accent)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
