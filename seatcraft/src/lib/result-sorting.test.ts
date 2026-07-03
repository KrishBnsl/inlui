import assert from "node:assert/strict";
import test from "node:test";
import { RESULT_SORT_LABELS, sortResults } from "./result-sorting.ts";
import {
  filterByBucket,
  formatChoiceName,
  getTopBucketChoice,
  getTierForResult,
  inferBucket,
  type PredictionResult,
  type RecommendationBucket,
} from "./types.ts";

function result(
  id: string,
  probability: number,
  recommendationScore: number,
  cutoff: number,
  instituteName = id,
  programName = "Computer Science and Engineering",
  safetyScore = probability,
  recommendationBucket?: RecommendationBucket
): PredictionResult {
  return {
    id,
    institute_name: instituteName,
    institute_type: "NIT",
    program_name: programName,
    quota_applied: "AI",
    category: "OPEN",
    probability_percent: probability,
    projected_closing_rank: cutoff,
    recommendation_score: recommendationScore,
    safety_score: safetyScore,
    recommendation_bucket: recommendationBucket,
    historical_data: [],
  };
}

test("best fit sorts by recommendation score before probability", () => {
  const sorted = sortResults([
    result("safe", 99, 20, 80_000),
    result("fit", 62, 88, 24_000),
  ]);

  assert.equal(sorted[0].id, "fit");
});

test("highest probability is explicit opt-in", () => {
  const sorted = sortResults(
    [
      result("safe", 99, 20, 80_000),
      result("fit", 62, 88, 24_000),
    ],
    "highest_probability"
  );

  assert.equal(sorted[0].id, "safe");
});

test("most competitive cutoff sorts by lower projected cutoff", () => {
  const sorted = sortResults(
    [
      result("loose", 95, 70, 90_000),
      result("tight", 45, 60, 12_000),
    ],
    "most_competitive"
  );

  assert.equal(sorted[0].id, "tight");
});

test("safest backup sorts by safety score", () => {
  const sorted = sortResults(
    [
      result("probable", 95, 70, 90_000, "NIT B", "Civil Engineering", 0.65),
      result("safest", 90, 65, 80_000, "NIT A", "Mechanical Engineering", 0.98),
    ],
    "safest_backup"
  );

  assert.equal(sorted[0].id, "safest");
});

test("branch name sorts alphabetically after institute modes", () => {
  const sorted = sortResults(
    [
      result("z", 80, 80, 30_000, "NIT Z", "Mechanical Engineering"),
      result("a", 80, 80, 30_000, "NIT A", "Artificial Intelligence"),
    ],
    "branch_name"
  );

  assert.equal(sorted[0].id, "a");
});

test("all expected sort labels are present", () => {
  assert.deepEqual(Object.keys(RESULT_SORT_LABELS), [
    "best_fit",
    "highest_probability",
    "most_competitive",
    "safest_backup",
    "institute_name",
    "branch_name",
  ]);
});

test("odds filters use backend recommendation buckets", () => {
  const rows = [
    result("safe", 95, 70, 60_000, "Safe College", "Civil Engineering", 0.9, "safe_backup"),
    result("target", 65, 90, 28_000, "Target College", "Electrical Engineering", 0.2, "best_realistic"),
    result("reach", 30, 80, 20_000, "Reach College", "Computer Science", 0, "ambitious_reach"),
  ];

  assert.deepEqual(filterByBucket(rows, "safe").map((row) => row.id), ["safe"]);
  assert.deepEqual(filterByBucket(rows, "target").map((row) => row.id), ["target"]);
  assert.deepEqual(filterByBucket(rows, "reach").map((row) => row.id), ["reach"]);
  assert.deepEqual(filterByBucket(rows, "ALL").map((row) => row.id), ["safe", "target", "reach"]);
});

test("best reach card and reach table use the same bucket source", () => {
  const rows = [
    result("reach-low", 35, 55, 22_000, "Reach B", "Mechanical Engineering", 0, "ambitious_reach"),
    result("reach-best", 30, 82, 20_000, "Reach A", "Computer Science", 0, "ambitious_reach"),
    result("target", 60, 90, 30_000, "Target", "Electrical Engineering", 0.2, "best_realistic"),
  ];

  const tableReach = filterByBucket(rows, "reach");
  const cardReach = getTopBucketChoice(rows, "ambitious_reach");

  assert.equal(cardReach?.id, "reach-best");
  assert.equal(tableReach.some((row) => row.id === cardReach?.id), true);
  assert.equal(formatChoiceName(cardReach), "Reach A — Computer Science");
});

test("reach filter supports unbucketed legacy results by probability tier", () => {
  const rows = [
    result("safe", 95, 70, 60_000, "Safe College", "Civil Engineering"),
    result("target", 65, 90, 28_000, "Target College", "Electrical Engineering"),
    result("reach", 30, 80, 20_000, "Reach College", "Computer Science"),
  ];

  const tableReach = filterByBucket(rows, "reach");
  const cardReach = getTopBucketChoice(rows, "ambitious_reach");

  assert.deepEqual(tableReach.map((row) => row.id), ["reach"]);
  assert.equal(cardReach?.id, "reach");
});

test("best reach card does not show a fake top-upgrade fallback", () => {
  const rows = [
    result("safe", 95, 70, 60_000, "Safe College", "Civil Engineering", 0.9, "safe_backup"),
    result("target", 65, 90, 28_000, "Target College", "Electrical Engineering", 0.2, "best_realistic"),
  ];

  const tableReach = filterByBucket(rows, "reach");
  const cardReach = getTopBucketChoice(rows, "ambitious_reach");

  assert.deepEqual(tableReach, []);
  assert.equal(formatChoiceName(cardReach, "No realistic reach found"), "No realistic reach found");
});

test("infer bucket falls back through confidence label and probability", () => {
  assert.equal(inferBucket({ ...result("safe", 20, 1, 10_000), confidence_label: "Safe" }), "safe_backup");
  assert.equal(inferBucket({ ...result("target", 20, 1, 10_000), confidence_label: "Moderate" }), "best_realistic");
  assert.equal(inferBucket({ ...result("reach", 80, 1, 10_000), confidence_label: "Ambitious" }), "ambitious_reach");
  assert.equal(inferBucket(result("prob-reach", 30, 1, 10_000)), "ambitious_reach");
});

test("visible tier state prefers backend bucket over raw probability", () => {
  const realisticReach = result(
    "reach",
    55,
    80,
    20_000,
    "Reach College",
    "Computer Science",
    0,
    "ambitious_reach"
  );

  assert.equal(getTierForResult(realisticReach), "reach");
});

test("hundred-row result set filters to twenty realistic reach rows", () => {
  const rows = [
    ...Array.from({ length: 30 }, (_, index) =>
      result(`safe-${index}`, 90, 90 - index, 60_000, `Safe College ${index}`, "Civil Engineering", 0.9, "safe_backup")
    ),
    ...Array.from({ length: 50 }, (_, index) =>
      result(`target-${index}`, 55, 80 - index, 3_100, `Target College ${index}`, "Electrical Engineering", 0.2, "best_realistic")
    ),
    ...Array.from({ length: 20 }, (_, index) =>
      result(`reach-${index}`, 10 + index, 70 - index, 2_500, `Reach College ${index}`, "Computer Science", 0, "ambitious_reach")
    ),
  ];

  const allRows = filterByBucket(rows, "ALL");
  const reachRows = filterByBucket(rows, "reach");
  const cardReach = getTopBucketChoice(rows, "ambitious_reach");

  assert.equal(allRows.length, 100);
  assert.equal(reachRows.length, 20);
  assert.equal(reachRows.every((row) => row.recommendation_bucket === "ambitious_reach"), true);
  assert.equal(reachRows.every((row) => row.probability_percent > 0), true);
  assert.equal(cardReach?.id, reachRows[0].id);
});
