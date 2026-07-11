import { expect, test, type Page } from "@playwright/test";

const mlHealth = {
  status: "ok",
  artifacts_loaded: true,
  universe_size: 3,
  artifact_status: { model: "loaded", universe: "loaded" },
};

const predictionResponse = {
  total_options: 3,
  safest_choice: "NIT Test Institute — Civil Engineering",
  top_upgrade: "IIIT Target Institute — Electronics Engineering",
  most_ambitious: "IIT Reach Institute — Mechanical Engineering",
  data_cutoff: "2024",
  model_version: "pre-counselling-ridge-v1",
  prediction_year: 2025,
  interval_label: "90% prediction interval",
  decision_support_disclaimer:
    "Decision support only; the uncalibrated admission-likelihood estimate is not an individual outcome probability or admission guarantee.",
  bucket_counts: {
    safe_backup: 1,
    best_realistic: 1,
    ambitious_reach: 1,
    unlikely_reach: 0,
  },
  candidate_counts: {
    universe: 300,
    after_profile_filters: 150,
    after_rank_routing: 120,
    scored: 120,
    eligible_after_reach_filter: 105,
    returned: 3,
  },
  returned_bucket_counts: {
    safe_backup: 1,
    best_realistic: 1,
    ambitious_reach: 1,
    unlikely_reach: 0,
  },
  results: [
    {
      id: "safe-nit",
      institute_name: "NIT Test Institute",
      institute_type: "NIT",
      program_name: "Civil Engineering",
      quota_applied: "AI",
      category: "OPEN",
      probability_percent: 92,
      admission_probability: 0.92,
      projected_closing_rank: 9_800,
      uncertainty_lower: 8_400,
      uncertainty_upper: 11_300,
      recommendation_score: 66,
      recommendation_bucket: "safe_backup",
      rank_used: 3_000,
      rank_type_used: "JEE_MAIN",
      historical_data: [
        { year: 2023, opening_rank: 4_500, closing_rank: 9_500 },
        { year: 2024, opening_rank: 4_700, closing_rank: 9_800 },
      ],
    },
    {
      id: "target-iiit",
      institute_name: "IIIT Target Institute",
      institute_type: "IIIT",
      program_name: "Electronics Engineering",
      quota_applied: "AI",
      category: "OPEN",
      probability_percent: 58,
      admission_probability: 0.58,
      projected_closing_rank: 3_600,
      uncertainty_lower: 3_100,
      uncertainty_upper: 4_200,
      recommendation_score: 91,
      recommendation_bucket: "best_realistic",
      rank_used: 3_000,
      rank_type_used: "JEE_MAIN",
      historical_data: [],
    },
    {
      id: "reach-iit",
      institute_name: "IIT Reach Institute",
      institute_type: "IIT",
      program_name: "Mechanical Engineering",
      quota_applied: "AI",
      category: "OPEN",
      probability_percent: 24,
      admission_probability: 0.24,
      projected_closing_rank: 6_100,
      uncertainty_lower: 5_400,
      uncertainty_upper: 7_200,
      recommendation_score: 79,
      recommendation_bucket: "ambitious_reach",
      rank_used: 6_700,
      rank_type_used: "JEE_ADVANCED",
      historical_data: [],
    },
  ],
};

async function fulfillJson(
  route: Parameters<Parameters<Page["route"]>[1]>[0],
  body: unknown,
  status = 200
) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockMlHealth(page: Page) {
  await page.route("**/mock-ml/api/v1/health", (route) => fulfillJson(route, mlHealth));
}

async function submitProfile(page: Page) {
  await page.getByLabel("JEE Main rank").fill("3000");
  await page.getByLabel("JEE Advanced rank").fill("6700");
  await page.getByLabel("Home state").selectOption("Delhi");
  await page.getByRole("button", { name: "Calculate options" }).click();
  await expect(page.getByRole("heading", { name: "Model recommendations" })).toBeVisible();
}

test("submits the NEXT_PUBLIC_ML_URL predict contract and renders transparent results", async ({
  page,
}) => {
  let payload: Record<string, unknown> | undefined;
  await mockMlHealth(page);
  await page.route("**/mock-ml/api/v1/predict", async (route) => {
    payload = route.request().postDataJSON();
    await new Promise((resolve) => setTimeout(resolve, 1_200));
    await fulfillJson(route, predictionResponse);
  });

  await page.goto("/");
  await page.getByLabel("JEE Main rank").fill("3000");
  await page.getByLabel("JEE Advanced rank").fill("6700");
  await page.getByLabel("Home state").selectOption("Delhi");
  await page.getByRole("button", { name: "Calculate options" }).click();

  await expect(page.getByText("Waiting for the ML inference service")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Model recommendations" })).toBeVisible();
  expect(payload).toEqual(
    expect.objectContaining({
      main_rank: 3000,
      advanced_rank: 6700,
      home_state: "Delhi",
      category: "OPEN",
      gender: "Gender-Neutral",
      is_pwd: false,
      round: 5,
      top_n: 100,
      sort_mode: "best_fit",
    })
  );

  const nitRow = page.getByTestId("result-row-safe-nit");
  const iitRow = page.getByTestId("result-row-reach-iit");
  await expect(nitRow).toContainText("NIT Test Institute");
  await expect(nitRow).toContainText("JEE Main");
  await expect(nitRow).toContainText("8,400–11,300");
  await expect(iitRow).toContainText("JEE Advanced");
  await expect(iitRow).toContainText("6,700");
  await expect(page.getByText("pre-counselling-ridge-v1")).toBeVisible();
  await expect(page.getByText("2024", { exact: true })).toBeVisible();
  await expect(page.getByText("2025", { exact: true })).toBeVisible();
  await expect(page.getByText("Heuristic, not probability").first()).toBeVisible();
  await expect(
    page.getByText(
      "Decision support only; the uncalibrated admission-likelihood estimate is not an individual outcome probability or admission guarantee."
    )
  ).toBeVisible();
  await expect(page.getByLabel("View the SeatCraft RAG branch on GitHub")).toHaveAttribute(
    "href",
    "https://github.com/KrishBnsl/inlui/tree/RAG"
  );
});

test("filters and sorts backend buckets and persists the choice list locally", async ({ page }) => {
  await mockMlHealth(page);
  await page.route("**/mock-ml/api/v1/predict", (route) =>
    fulfillJson(route, predictionResponse)
  );
  await page.goto("/");
  await submitProfile(page);

  await page.getByTestId("bucket-filter-safe").click();
  await expect(page.getByTestId("result-row-safe-nit")).toBeVisible();
  await expect(page.getByTestId("result-row-target-iiit")).toHaveCount(0);

  await page.getByTestId("bucket-filter-target").click();
  await expect(page.getByTestId("result-row-target-iiit")).toBeVisible();
  await expect(page.getByTestId("result-row-safe-nit")).toHaveCount(0);

  await page.getByTestId("bucket-filter-reach").click();
  await expect(page.getByTestId("result-row-reach-iit")).toBeVisible();
  await expect(page.getByTestId("result-row-target-iiit")).toHaveCount(0);

  await page.getByTestId("bucket-filter-all").click();
  await page.getByLabel("Sort results").selectOption("highest_probability");
  const rows = page.locator("tbody tr[data-testid^='result-row-']");
  await expect(rows.first()).toContainText("NIT Test Institute");

  await page
    .getByRole("button", {
      name: "Save IIIT Target Institute Electronics Engineering to choice list",
    })
    .click();
  await expect(page.getByText("1 saved options")).toBeVisible();
  await page.getByLabel("Counselling notes").fill("Synthetic E2E note");
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("seatcraft.choiceIds")))
    .toBe('["target-iiit"]');
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("seatcraft.choiceNote")))
    .toBe("Synthetic E2E note");

  await page.reload();
  await submitProfile(page);
  await expect(page.getByText("1 saved options")).toBeVisible();
  await expect(page.getByLabel("Counselling notes")).toHaveValue("Synthetic E2E note");

  await page.evaluate(() => localStorage.setItem("seatcraft.choiceIds", "{}"));
  await page.reload();
  await expect(
    page.getByRole("heading", {
      name: "Build a JoSAA list with uncertainty you can inspect.",
    })
  ).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("seatcraft.choiceIds")))
    .toBe("[]");
});

test("advisor checks RAG health, uploads a source, asks with ML context, and displays retrieved excerpts", async ({
  page,
}) => {
  let healthCalls = 0;
  let askPayload: Record<string, unknown> | undefined;
  let contextPayload: Record<string, unknown> | undefined;

  await mockMlHealth(page);
  await page.route("**/mock-ml/api/v1/predict", (route) =>
    fulfillJson(route, predictionResponse)
  );
  await page.route("**/mock-ml/api/v1/chat-context", (route) => {
    contextPayload = route.request().postDataJSON();
    return fulfillJson(route, {
      context_block: "Student rank context from the ML service.",
      key_facts: { main_rank: 3000 },
    });
  });
  await page.route("**/mock-rag/health", (route) => {
    healthCalls += 1;
    return fulfillJson(route, {
      status: "ok",
      ready: true,
      google_api_key_configured: true,
      missing: [],
      embedding_provider_ready: true,
      chat_provider_ready: true,
      vector_store_available: false,
      chunk_count: 0,
      degraded_reasons: [],
      configured_models: {
        embedding: "synthetic-embedding",
        chat: "synthetic-chat",
      },
    });
  });
  await page.route("**/mock-rag/api/rag/upload", (route) =>
    fulfillJson(route, { ok: true, chunks_added: 2, source: "synthetic-guide.pdf" })
  );
  await page.route("**/mock-rag/api/rag/ask", (route) => {
    askPayload = route.request().postDataJSON();
    return fulfillJson(route, {
      answer: "Use the current official schedule and keep a balanced list.",
      sources: [
        {
          source: "synthetic-guide.pdf",
          excerpt: "Synthetic fixture excerpt for browser contract testing.",
        },
      ],
    });
  });

  await page.goto("/");
  await submitProfile(page);
  await page.getByRole("button", { name: "Open counselling advisor" }).click();
  await expect(page.getByTestId("rag-health-status")).toContainText("Gemini configured");
  expect(healthCalls).toBe(1);

  await page.locator(`input[type="file"][accept=".pdf,.png,.jpg,.jpeg,.webp"]`).setInputFiles({
    name: "synthetic-guide.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4 synthetic browser fixture"),
  });
  await expect(page.getByText("synthetic-guide.pdf").first()).toBeVisible();
  await expect(page.getByText(/2 source chunks added/)).toBeVisible();

  await page.getByLabel("Ask the counselling advisor").fill("How should I order these choices?");
  await page.getByRole("button", { name: "Send question to counselling advisor" }).click();
  await expect(page.getByText("Use the current official schedule and keep a balanced list.")).toBeVisible();
  expect(contextPayload).toEqual(
    expect.objectContaining({ question: "How should I order these choices?" })
  );
  expect(askPayload).toEqual(
    expect.objectContaining({
      question: "How should I order these choices?",
      ml_context: "Student rank context from the ML service.",
    })
  );
  await page.getByRole("button", { name: "1 source" }).click();
  await expect(
    page.getByText("Synthetic fixture excerpt for browser contract testing.")
  ).toBeVisible();
});

test("invalid intake and backend failures stay visible without fake recommendations", async ({
  page,
}) => {
  let predictionCalls = 0;
  await mockMlHealth(page);
  await page.route("**/mock-ml/api/v1/predict", (route) => {
    predictionCalls += 1;
    return fulfillJson(route, { detail: "Prediction service temporarily unavailable." }, 503);
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Calculate options" }).click();
  await expect(page.getByText("JEE Main rank is required.")).toBeVisible();
  await expect(page.getByText("Home state is required for quota routing.")).toBeVisible();
  expect(predictionCalls).toBe(0);

  await page.getByLabel("JEE Main rank").fill("1200001");
  await page.getByLabel("JEE Advanced rank").fill("50001");
  await page.getByLabel("Home state").selectOption("Delhi");
  await page.getByRole("button", { name: "Calculate options" }).click();
  await expect(page.getByText("JEE Main rank must be 12,00,000 or below.")).toBeVisible();
  await expect(page.getByText("JEE Advanced rank must be 50,000 or below.")).toBeVisible();
  expect(predictionCalls).toBe(0);

  await page.getByLabel("JEE Main rank").fill("3000");
  await page.getByLabel("JEE Advanced rank").fill("6700");
  await page.getByRole("button", { name: "Calculate options" }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "Prediction service temporarily unavailable." })
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Model recommendations" })).toHaveCount(0);
  await expect(page.getByText("NIT Test Institute")).toHaveCount(0);
  expect(predictionCalls).toBe(1);
});

test("RAG health and ask failures are explicit", async ({ page }) => {
  await page.route("**/mock-rag/health", (route) =>
    fulfillJson(route, { detail: "unavailable" }, 503)
  );
  await page.route("**/mock-rag/api/rag/ask", (route) =>
    fulfillJson(route, { detail: "Advisor upstream unavailable." }, 503)
  );

  await page.goto("/");
  await page.getByRole("button", { name: "Open counselling advisor" }).click();
  await expect(page.getByTestId("rag-health-status")).toContainText("RAG service unavailable.");
  await page.getByLabel("Ask the counselling advisor").fill("Is the advisor available?");
  await page.getByRole("button", { name: "Send question to counselling advisor" }).click();
  await expect(page.getByText("Error: Advisor upstream unavailable.")).toBeVisible();
});

test("configured but degraded RAG reports the backend reason accurately", async ({ page }) => {
  await page.route("**/mock-rag/health", (route) =>
    fulfillJson(route, {
      status: "degraded",
      ready: false,
      google_api_key_configured: true,
      missing: [],
      embedding_provider_ready: true,
      chat_provider_ready: true,
      vector_store_available: false,
      chunk_count: 0,
      degraded_reasons: ["FAISS runtime is unavailable"],
      configured_models: {
        embedding: "synthetic-embedding",
        chat: "synthetic-chat",
      },
    })
  );

  await page.goto("/");
  await page.getByRole("button", { name: "Open counselling advisor" }).click();
  await expect(page.getByTestId("rag-health-status")).toContainText(
    "RAG degraded: FAISS runtime is unavailable."
  );
});

test("mobile results expose the same filters and save control", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockMlHealth(page);
  await page.route("**/mock-ml/api/v1/predict", (route) =>
    fulfillJson(route, predictionResponse)
  );
  await page.goto("/");
  await submitProfile(page);
  await expect(page.getByTestId("mobile-result-safe-nit")).toBeVisible();
  await page.getByTestId("bucket-filter-reach").click();
  await expect(page.getByTestId("mobile-result-reach-iit")).toBeVisible();
  await expect(page.getByTestId("mobile-result-safe-nit")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Save IIT Reach Institute Mechanical Engineering" })
    .click();
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("seatcraft.choiceIds")))
    .toBe('["reach-iit"]');
});
