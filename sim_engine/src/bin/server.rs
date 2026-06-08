//! HTTP API server — exposes the JoSAA Monte Carlo engine over JSON.
//!
//! POST /api/simulate  →  SimulateRequest  →  SimulateResponse
//! GET  /api/health    →  200 OK

use std::{collections::{HashMap, HashSet}, env, sync::Arc};

use axum::{
    extract::{Multipart, State},
    http::{HeaderValue, Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use sim_engine::predictor::{
    db,
    engine::PredictorEngine,
    normalization::{CohortRegistry, ExamCohort},
    quota_matrix::{HorizontalFlags, UserProfile},
};
use sim_engine::enumerations::enums::{category, quota, IndianState};
use sim_engine::rag::{self, RagService};

// ─── RAG request / response types ───────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct AskRequest {
    question: String,
    /// Optional base64-encoded image the user attaches to their question.
    image_b64: Option<String>,
}

// ─── JSON Request/Response types (mirrors the TypeScript frontend types) ──────

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct SimulateRequest {
    main_rank: u32,
    advanced_rank: Option<u32>,
    category: String,
    home_state: String,
    gender: String,
    is_pwd: bool,
}

#[derive(Debug, Serialize)]
struct HistoricalPoint {
    year: u16,
    opening_rank: u32,
    closing_rank: u32,
}

#[derive(Debug, Serialize)]
struct PredictionResult {
    id: String,
    institute_name: String,
    institute_type: String,
    program_name: String,
    quota_applied: String,
    category: String,
    probability_percent: f64,
    projected_closing_rank: u32,
    uses_cold_start_proxy: bool,
    historical_data: Vec<HistoricalPoint>,
}

#[derive(Debug, Serialize)]
struct SimulateResponse {
    total_options: usize,
    safest_choice: String,
    top_upgrade: String,
    results: Vec<PredictionResult>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

// ─── Shared application state ─────────────────────────────────────────────────

struct AppState {
    engine_josaa: PredictorEngine,  // JoSAA engine (NITs, IIITs, GFTIs via JEE Main)
    engine_iit: PredictorEngine,    // JoSAA IIT engine (via JEE Advanced)
    rag: Arc<RagService>,           // RAG pipeline (in-process, no second service)
}

// ─── Parse helpers ────────────────────────────────────────────────────────────

fn parse_category(s: &str) -> category {
    match s {
        "OBC-NCL" | "OBC" => category::OBC,
        "SC"               => category::SC,
        "ST"               => category::ST,
        "EWS"              => category::EWS,
        _                  => category::General, // "OPEN"
    }
}

fn parse_state(s: &str) -> Option<IndianState> {
    match s {
        "Andhra Pradesh"       => Some(IndianState::AndhraPradesh),
        "Arunachal Pradesh"    => Some(IndianState::ArunachalPradesh),
        "Assam"                => Some(IndianState::Assam),
        "Bihar"                => Some(IndianState::Bihar),
        "Chhattisgarh"         => Some(IndianState::Chhattisgarh),
        "Goa"                  => Some(IndianState::Goa),
        "Gujarat"              => Some(IndianState::Gujarat),
        "Haryana"              => Some(IndianState::Haryana),
        "Himachal Pradesh"     => Some(IndianState::HimachalPradesh),
        "Jharkhand"            => Some(IndianState::Jharkhand),
        "Karnataka"            => Some(IndianState::Karnataka),
        "Kerala"               => Some(IndianState::Kerala),
        "Madhya Pradesh"       => Some(IndianState::MadhyaPradesh),
        "Maharashtra"          => Some(IndianState::Maharashtra),
        "Manipur"              => Some(IndianState::Manipur),
        "Meghalaya"            => Some(IndianState::Meghalaya),
        "Mizoram"              => Some(IndianState::Mizoram),
        "Nagaland"             => Some(IndianState::Nagaland),
        "Odisha"               => Some(IndianState::Odisha),
        "Punjab"               => Some(IndianState::Punjab),
        "Rajasthan"            => Some(IndianState::Rajasthan),
        "Sikkim"               => Some(IndianState::Sikkim),
        "Tamil Nadu"           => Some(IndianState::TamilNadu),
        "Telangana"            => Some(IndianState::Telangana),
        "Tripura"              => Some(IndianState::Tripura),
        "Uttar Pradesh"        => Some(IndianState::UttarPradesh),
        "Uttarakhand"          => Some(IndianState::Uttarakhand),
        "West Bengal"          => Some(IndianState::WestBengal),
        "Delhi"                => Some(IndianState::Delhi),
        "Chandigarh"           => Some(IndianState::Chandigarh),
        _                      => None,
    }
}

/// Map institute type string from DB to display name.
fn institute_type_label(name: &str) -> &'static str {
    let n = name.to_ascii_uppercase();
    if n.contains("INDIAN INSTITUTE OF TECHNOLOGY") { "IIT" }
    else if n.contains("NATIONAL INSTITUTE OF TECHNOLOGY") { "NIT" }
    else if n.contains("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY") { "IIIT" }
    else { "GFTI" }
}

// ─── JEE cohort registry (candidate pool sizes by year) ──────────────────────

fn jee_main_cohorts() -> CohortRegistry {
    CohortRegistry::new([
        ExamCohort { year: 2018, total_candidates: 1_043_739 },
        ExamCohort { year: 2019, total_candidates: 1_041_804 },
        ExamCohort { year: 2020, total_candidates:   858_273 },
        ExamCohort { year: 2021, total_candidates: 1_114_000 },
        ExamCohort { year: 2022, total_candidates: 1_048_012 },
        ExamCohort { year: 2023, total_candidates: 1_145_000 },
        ExamCohort { year: 2024, total_candidates: 1_180_000 },
        ExamCohort { year: 2025, total_candidates: 1_200_000 }, // projection
        ExamCohort { year: 2026, total_candidates: 1_250_000 }, // target year
    ])
}

fn jee_advanced_cohorts() -> CohortRegistry {
    CohortRegistry::new([
        ExamCohort { year: 2018, total_candidates: 155_158 },
        ExamCohort { year: 2019, total_candidates: 161_319 },
        ExamCohort { year: 2020, total_candidates: 150_838 },
        ExamCohort { year: 2021, total_candidates: 141_699 },
        ExamCohort { year: 2022, total_candidates: 155_538 },
        ExamCohort { year: 2023, total_candidates: 189_744 },
        ExamCohort { year: 2024, total_candidates: 180_200 },
        ExamCohort { year: 2025, total_candidates: 185_000 }, // projection
        ExamCohort { year: 2026, total_candidates: 190_000 }, // target year
    ])
}

// ─── Routes ───────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

// ─── RAG routes ──────────────────────────────────────────────────────────────

/// POST /api/rag/upload — multipart form with a `file` field (PDF or image).
async fn rag_upload(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_string();
        if field_name != "file" {
            continue;
        }

        let filename = field
            .file_name()
            .unwrap_or("upload")
            .to_string();

        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": format!("Failed to read file: {e}") })),
                )
                    .into_response();
            }
        };

        return match rag::ingestion::ingest_document(&state.rag, &filename, &bytes).await {
            Ok(count) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "ok": true,
                    "chunks_added": count,
                    "source": filename,
                })),
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response(),
        };
    }

    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": "No 'file' field found in multipart body" })),
    )
        .into_response()
}

/// POST /api/rag/ask — JSON body { question, image_b64? }.
async fn rag_ask(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AskRequest>,
) -> impl IntoResponse {
    match rag::retrieval::answer(&state.rag, &req.question, req.image_b64.as_deref()).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// GET /api/rag/status — returns chunk count and indexed document names.
async fn rag_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let store = state.rag.store.read().await;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "chunk_count": store.chunk_count(),
            "documents": store.documents(),
        })),
    )
        .into_response()
}


async fn simulate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SimulateRequest>,
) -> impl IntoResponse {

    let cat = parse_category(&req.category);
    let homestate = match parse_state(&req.home_state) {
        Some(s) => s,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Unknown state: {}", req.home_state) })),
            ).into_response();
        }
    };
    let is_female = req.gender.contains("Female") || req.gender.contains("female");

    let horizontal = HorizontalFlags {
        female: is_female,
        defence: false,
        pwd: req.is_pwd,
    };

    let mut all_results: Vec<(String, PredictionResult)> = Vec::new();

    // ── NIT/IIIT/GFTI predictions via JEE Main rank ──────────────────────────
    {
        // A student competes for HS quota in their home state, AI for everything else.
        // We run predict_for_user twice — once with HS profile, once with AI profile —
        // the engine's routing layer will return only assets that match each profile.
        for q in [quota::HomeState, quota::OtherState] {
            let user = UserProfile { category: cat, quota: q, homestate, horizontal };
            match state.engine_josaa.predict_for_user(&user, req.main_rank) {
                Ok(predictions) => {
                    for pred in predictions {
                        let quota_label = match q {
                            quota::HomeState  => "HS",
                            quota::OtherState => "AI",
                        };

                        // Reconstruct historical data from the raw yearly_cutoffs on the asset
                        let asset = state.engine_josaa.matrix.find_by_key(&pred.key);
                        let mut historical = asset
                            .map(|a| {
                                let mut pts: Vec<HistoricalPoint> = a.yearly_cutoffs.iter()
                                    .map(|(yr, closing)| HistoricalPoint {
                                        year: *yr,
                                        opening_rank: closing.saturating_sub(closing / 5), // approx
                                        closing_rank: *closing,
                                    })
                                    .collect();
                                pts.sort_by_key(|p| p.year);
                                pts
                            })
                            .unwrap_or_default();

                        // Use a stable unique key for dedup: institute + branch + quota + gender flag
                        let gender_tag = if horizontal.female { "F" } else { "N" };
                        let dedup_key = format!("{}-{}-{}-{}", pred.key.institute, pred.key.branch, quota_label, gender_tag);

                        all_results.push((dedup_key, PredictionResult {
                            id: String::new(), // filled after dedup
                            institute_type: institute_type_label(&pred.key.institute).to_string(),
                            institute_name: pred.key.institute.clone(),
                            program_name: pred.key.branch.clone(),
                            quota_applied: quota_label.to_string(),
                            category: req.category.clone(),
                            probability_percent: (pred.probability_percent * 10.0).round() / 10.0,
                            projected_closing_rank: pred.projected_cutoff_rank,
                            uses_cold_start_proxy: pred.uses_cold_start_proxy,
                            historical_data: historical,
                        }));
                    }
                }
                Err(e) => {
                    tracing::warn!("JoSAA prediction error (quota={:?}): {:?}", q, e);
                }
            }
        }
    }

    // ── IIT predictions via JEE Advanced rank ─────────────────────────────────
    if let Some(adv_rank) = req.advanced_rank {
        let user = UserProfile {
            category: cat,
            quota: quota::OtherState, // IITs are all-India only
            homestate,
            horizontal,
        };
        match state.engine_iit.predict_for_user(&user, adv_rank) {
            Ok(predictions) => {
                for pred in predictions {
                    let asset = state.engine_iit.matrix.find_by_key(&pred.key);
                    let historical = asset
                        .map(|a| {
                            let mut pts: Vec<HistoricalPoint> = a.yearly_cutoffs.iter()
                                .map(|(yr, closing)| HistoricalPoint {
                                    year: *yr,
                                    opening_rank: closing.saturating_sub(closing / 5),
                                    closing_rank: *closing,
                                })
                                .collect();
                            pts.sort_by_key(|p| p.year);
                            pts
                        })
                        .unwrap_or_default();

                    let gender_tag = if horizontal.female { "F" } else { "N" };
                    let dedup_key = format!("{}-{}-AI-{}", pred.key.institute, pred.key.branch, gender_tag);

                    all_results.push((dedup_key, PredictionResult {
                        id: String::new(),
                        institute_type: "IIT".to_string(),
                        institute_name: pred.key.institute.clone(),
                        program_name: pred.key.branch.clone(),
                        quota_applied: "AI".to_string(),
                        category: req.category.clone(),
                        probability_percent: (pred.probability_percent * 10.0).round() / 10.0,
                        projected_closing_rank: pred.projected_cutoff_rank,
                        uses_cold_start_proxy: pred.uses_cold_start_proxy,
                        historical_data: historical,
                    }));
                }
            }
            Err(e) => {
                tracing::warn!("IIT prediction error: {:?}", e);
            }
        }
    }

    // Deduplicate using a HashSet on the stable dedup_key (handles non-consecutive duplicates)
    let mut seen: HashSet<String> = HashSet::new();
    let mut unique: Vec<PredictionResult> = all_results
        .into_iter()
        .filter_map(|(key, result)| {
            if seen.insert(key) { Some(result) } else { None }
        })
        .collect();

    // Sort by probability descending
    unique.sort_by(|a, b| {
        b.probability_percent
            .partial_cmp(&a.probability_percent)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign stable sequential IDs now that order is finalised
    let all_results: Vec<PredictionResult> = unique
        .into_iter()
        .enumerate()
        .map(|(i, mut r)| { r.id = i.to_string(); r })
        .collect();
    let total = all_results.len();
    let safest = all_results.first().map(|r| r.institute_name.clone()).unwrap_or_default();
    let upgrade_idx = (total as f64 * 0.33) as usize;
    let upgrade = all_results.get(upgrade_idx).map(|r| r.institute_name.clone()).unwrap_or_default();

    let response = SimulateResponse {
        total_options: total,
        safest_choice: safest,
        top_upgrade: upgrade,
        results: all_results,
    };

    (StatusCode::OK, Json(response)).into_response()
}

// ─── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let port = env::var("PORT").unwrap_or_else(|_| "8080".into());

    info!("Connecting to database…");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .expect("Cannot connect to database");

    info!("Loading cutoff data from database (single fetch)…");
    let all_rows: Vec<sim_engine::predictor::db::HistoricalCutoff> =
        db::fetch_cutoffs_for_simulation(&pool)
            .await
            .expect("Failed to load cutoff data from database");

    info!("Building JoSAA engine (NITs/IIITs/GFTIs)…");
    // JoSAA engine covers NITs, IIITs, GFTIs, and non-IIT institutes
    let josaa_rows: Vec<_> = all_rows.iter()
        .filter(|r| !r.institute_name.to_uppercase().contains("INDIAN INSTITUTE OF TECHNOLOGY"))
        .cloned()
        .collect();
    let engine_josaa = PredictorEngine::from_rows(
        josaa_rows,
        2026,
        1_250_000,
        jee_main_cohorts(),
        HashMap::new(),
    );

    info!("Building IIT engine (JEE Advanced)…");
    // IIT engine covers only IIT rows — smaller matrix, faster routing
    let iit_rows: Vec<_> = all_rows.into_iter()
        .filter(|r| r.institute_name.to_uppercase().contains("INDIAN INSTITUTE OF TECHNOLOGY"))
        .collect();
    let engine_iit = PredictorEngine::from_rows(
        iit_rows,
        2026,
        190_000,
        jee_advanced_cohorts(),
        HashMap::new(),
    );


    info!("Initialising RAG service…");
    let rag_config = sim_engine::rag::config::RagConfig::from_env();
    if rag_config.api_key.is_empty() {
        tracing::warn!(
            "OPENAI_API_KEY is not set — RAG endpoints will return errors until it is configured."
        );
    }
    let rag = Arc::new(RagService::new(rag_config));

    let state = Arc::new(AppState { engine_josaa, engine_iit, rag });

    let cors = CorsLayer::new()
        .allow_origin("http://localhost:3000".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/simulate", post(simulate))
        .route("/api/rag/upload", post(rag_upload))
        .route("/api/rag/ask",    post(rag_ask))
        .route("/api/rag/status", get(rag_status))
        .layer(cors)
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("Server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
