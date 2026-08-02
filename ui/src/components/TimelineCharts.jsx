import { useId, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ResponsiveContainer,
} from "recharts";
import { classBands, priceTickLabel, priceTickAxisWidth } from "../timelineData.js";
import { formatDateTime } from "../format.js";

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function timeLabel(timestamp) {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "Unavailable" : timeFormatter.format(date);
}

function tooltipFormatter(value, name) {
  if (value === null || typeof value !== "number") return ["—", name];
  if (name === "BTC price") return [`$${value.toFixed(2)}`, name];
  return [value.toFixed(3), name];
}

// Shared axis geometry so the two stacked charts stay pixel-aligned: same
// left/right margins and the same YAxis width on both, regardless of which
// one shows tick labels. The width itself is computed per-render from the
// price domain (see priceTickAxisWidth) rather than fixed.
const CHART_MARGIN = { top: 8, right: 40, left: 0, bottom: 8 };

// BTC price (top) and spike prediction-vs-reality (bottom), stacked with a
// shared x-axis, range, and hover cursor (recharts syncId). No dual y-axis:
// price and score live on genuinely different scales, and forcing them onto
// one panel would visually imply a correlation the scales alone don't
// support.
export default function TimelineCharts({ points, from, to, complete, availableFrom }) {
  const summaryId = useId();
  const domain = useMemo(() => [Date.parse(from), Date.parse(to)], [from, to]);

  const prices = useMemo(
    () => points.map((p) => p.price).filter((v) => typeof v === "number"),
    [points],
  );
  const priceDomainSpan = prices.length >= 2 ? Math.max(...prices) - Math.min(...prices) : 0;
  const priceAxisWidth = priceTickAxisWidth(priceDomainSpan);
  // market_price is a nullable migrated column, so a whole window can legitimately
  // carry no price at all. Rendering a Line over all-nulls draws an empty axis
  // with no line and no explanation, which reads as a broken chart rather than
  // as absent data -- say so instead.
  const hasPrice = prices.length > 0;

  // A count, not a mystery: ungradeable stretches are shaded, and without a
  // number next to them there's no way to tell "a couple of rows" from "most
  // of this window".
  const ungradeableCount = useMemo(
    () => points.filter((p) => p.unavailable).length,
    [points],
  );

  const predictedBands = useMemo(() => classBands(points, "predicted"), [points]);
  const confirmedBands = useMemo(() => classBands(points, "confirmedSpike"), [points]);
  const pendingBands = useMemo(() => classBands(points, "pending"), [points]);
  const unavailableBands = useMemo(() => classBands(points, "unavailable"), [points]);

  const summary = `Timeline with ${points.length} points from ${timeLabel(domain[0])} to ${timeLabel(domain[1])}.`;

  return (
    <div className="timeline-charts">
      {!complete && (
        <p className="state-message timeline-coverage-note" role="status">
          Retained history begins {formatDateTime(availableFrom)} — this range is truncated.
        </p>
      )}

      <figure
        className="score-chart timeline-chart-price"
        aria-labelledby="timeline-price-heading"
        aria-describedby={summaryId}
      >
        <figcaption className="sr-only" id={summaryId}>
          {summary}
        </figcaption>
        <p className="eyebrow" id="timeline-price-heading">
          BTC price
        </p>
        {!hasPrice ? (
          <p className="state-message">
            No price recorded for this stretch of data. Price capture was added after
            these events were logged, so the model's scores below are still valid —
            there's just nothing to plot alongside them.
          </p>
        ) : (
        <div className="score-chart-visual" role="img" aria-label="BTC price over the selected range">
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={points} syncId="timeline" margin={CHART_MARGIN}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={domain}
                tick={false}
                axisLine={{ stroke: "var(--border)" }}
                tickLine={false}
              />
              <YAxis
                domain={["auto", "auto"]}
                tickFormatter={(value) => priceTickLabel(value, priceDomainSpan)}
                tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                width={priceAxisWidth}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-panel)",
                  border: "1px solid var(--border-strong)",
                  borderRadius: 4,
                  fontSize: 12,
                }}
                labelFormatter={(label) => `Time: ${timeLabel(label)}`}
                formatter={tooltipFormatter}
              />
              <Line
                type="monotone"
                dataKey="price"
                name="BTC price"
                stroke="var(--green)"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        )}
      </figure>

      <figure className="score-chart timeline-chart-prediction" aria-labelledby="timeline-prediction-heading">
        <p className="eyebrow" id="timeline-prediction-heading">
          Spike prediction vs. reality
        </p>
        <div
          className="score-chart-visual"
          role="img"
          aria-label="Model spike score and threshold, with predicted, confirmed, pending, and unavailable outcome regions"
        >
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={points} syncId="timeline" margin={CHART_MARGIN}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={domain}
                tickFormatter={timeLabel}
                tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                minTickGap={40}
              />
              <YAxis domain={[0, 1]} tick={{ fill: "var(--text-muted)", fontSize: 11 }} width={priceAxisWidth} />
              <Tooltip
                contentStyle={{
                  background: "var(--bg-panel)",
                  border: "1px solid var(--border-strong)",
                  borderRadius: 4,
                  fontSize: 12,
                }}
                labelFormatter={(label) => `Time: ${timeLabel(label)}`}
                formatter={tooltipFormatter}
              />
              {unavailableBands.map((band, index) => (
                <ReferenceArea
                  key={`unavailable-${band.x1}-${index}`}
                  x1={band.x1}
                  x2={band.x2}
                  fill="var(--red)"
                  fillOpacity={0.12}
                  stroke="var(--red)"
                  strokeOpacity={0.3}
                  strokeDasharray="2 2"
                  ifOverflow="extendDomain"
                />
              ))}
              {pendingBands.map((band, index) => (
                <ReferenceArea
                  key={`pending-${band.x1}-${index}`}
                  x1={band.x1}
                  x2={band.x2}
                  fill="var(--gray)"
                  fillOpacity={0.18}
                  stroke="none"
                  ifOverflow="extendDomain"
                />
              ))}
              {predictedBands.map((band, index) => (
                <ReferenceArea
                  key={`predicted-${band.x1}-${index}`}
                  x1={band.x1}
                  x2={band.x2}
                  fill="var(--accent-soft)"
                  stroke="none"
                  ifOverflow="extendDomain"
                />
              ))}
              {confirmedBands.map((band, index) => (
                <ReferenceArea
                  key={`confirmed-${band.x1}-${index}`}
                  x1={band.x1}
                  x2={band.x2}
                  fill="var(--green)"
                  fillOpacity={0.16}
                  stroke="none"
                  ifOverflow="extendDomain"
                />
              ))}
              <Line
                type="stepAfter"
                dataKey="tau"
                name="Alert threshold"
                stroke="var(--amber)"
                strokeWidth={1.5}
                strokeDasharray="5 4"
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="score"
                name="Score"
                stroke="var(--accent)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </figure>

      <p className="panel-meta timeline-legend">
        <span style={{ color: "var(--green)" }}>■</span> BTC price ·{" "}
        <span style={{ color: "var(--accent)" }}>■</span> Score ·{" "}
        <span style={{ color: "var(--amber)" }}>■</span> Alert threshold ·{" "}
        <span style={{ color: "var(--accent)" }}>▨</span> Model alerted ·{" "}
        <span style={{ color: "var(--green)" }}>▨</span> Spike really happened ·{" "}
        <span style={{ color: "var(--gray)" }}>▨</span> Waiting for the 60s result ·{" "}
        <span style={{ color: "var(--red)" }}>▨</span> Not gradeable
      </p>

      <p className="panel-meta timeline-howto">
        <strong>How to read this:</strong> the blue line is the model's confidence;
        when it crosses the dashed threshold the model is alerting, and that stretch
        shades blue. Green shading is where a spike genuinely happened. Blue over
        green means it called it right; blue with no green underneath was a false
        alarm; green with no blue was a miss.
        {ungradeableCount > 0 && (
          <>
            {" "}
            <span className="timeline-howto-count">
              {ungradeableCount} of {points.length}{" "}
              {points.length === 1 ? "point is" : "points are"} shaded red — those
              can't be honestly graded (the outcome never arrived, or the score
              landed after the answer was already known), so they're excluded from
              the metrics rather than guessed at.
            </span>
          </>
        )}
      </p>
    </div>
  );
}
