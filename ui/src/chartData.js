export function buildChartData(predictions) {
  return predictions
    .map((prediction) => ({
      timestamp: new Date(prediction.api_ts || prediction.feature_ts).getTime(),
      score: prediction.score,
    }))
    .filter(
      (point) =>
        Number.isFinite(point.timestamp) && typeof point.score === "number",
    )
    .sort((left, right) => left.timestamp - right.timestamp);
}
