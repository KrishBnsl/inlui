import assert from "node:assert/strict";
import test from "node:test";
import { RESULT_SORT_LABELS, sortResults } from "./result-sorting.ts";
import {
  filterByBucket,
  formatChoiceName,
  getTopBucketChoice,
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
