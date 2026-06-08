use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct HistoricalCutoff {
    pub institute_id: i32,
    pub program_id: i32,
    pub year: i32,
    pub round: i32,
    pub quota: String,
    pub category: String,
    pub gender: String,
    pub is_pwd: bool,
    pub is_defence: bool,
    pub opening_rank: i32,
    pub closing_rank: i32,
    
    // Virtual fields joined from the institutes/programs tables for our routing mapping
    pub institute_name: String,
    pub program_name: String,
    pub state: String,
}

pub async fn fetch_cutoffs_for_simulation(pool: &PgPool) -> Result<Vec<HistoricalCutoff>, sqlx::Error> {
    // No ORDER BY — results are aggregated into a HashMap in memory so DB ordering is unused overhead.
    sqlx::query_as::<_, HistoricalCutoff>(
        r#"
        SELECT 
            c.institute_id,
            c.program_id,
            c.year,
            c.round,
            c.quota,
            c.category,
            c.gender,
            c.is_pwd,
            c.is_defence,
            c.opening_rank,
            c.closing_rank,
            i.name as institute_name,
            p.name as program_name,
            i.state as state
        FROM josaa_cutoffs c
        JOIN institutes i ON c.institute_id = i.id
        JOIN programs p ON c.program_id = p.id
        "#
    )
    .fetch_all(pool)
    .await
}
