use csv::ReaderBuilder;
use dotenvy::dotenv;
use serde::Deserialize;
use sqlx::{postgres::PgPoolOptions, PgPool};
use std::{collections::HashMap, env, error::Error, fs::File};

#[derive(Debug, Deserialize)]
struct CsvRow {
    #[serde(rename = "Institute")]
    institute: String,
    #[serde(rename = "Academic Program Name")]
    program_name: String,
    #[serde(rename = "Quota")]
    quota: String,
    #[serde(rename = "Seat Type")]
    seat_type: String,
    #[serde(rename = "Gender")]
    gender: String,
    #[serde(rename = "Opening Rank")]
    opening_rank: Option<f64>, // Some rows have empty rank (supernumerary / preparatory seats)
    #[serde(rename = "Closing Rank")]
    closing_rank: Option<f64>,
    #[serde(rename = "Round")]
    round: i32,
    #[serde(rename = "Year")]
    year: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    println!("Creating tables...");
    setup_database(&pool).await?;

    println!("Parsing CSV...");
    let file = File::open("merged_jee_cutoff_2018_2025.csv")?;
    let mut rdr = ReaderBuilder::new().from_reader(file);

    let mut inst_cache: HashMap<String, i32> = HashMap::new();
    let mut prog_cache: HashMap<String, i32> = HashMap::new();
    let mut inst_code_counter = 1000;
    let mut prog_code_counter = 1000;

    let mut count = 0;

    for result in rdr.deserialize() {
        let record: CsvRow = result?;

        // Skip rows with no rank data (supernumerary / preparatory seats)
        let (Some(opening_rank), Some(closing_rank)) = (record.opening_rank, record.closing_rank) else {
            continue;
        };

        // 1. Get or create Institute
        let inst_id = if let Some(id) = inst_cache.get(&record.institute) {
            *id
        } else {
            let inst_type = if record.institute.contains("Indian Institute of Technology") {
                "IIT"
            } else if record.institute.contains("National Institute of Technology") {
                "NIT"
            } else if record.institute.contains("Indian Institute of Information Technology") {
                "IIIT"
            } else {
                "GFTI"
            };

            let row: (i32,) = sqlx::query_as(
                r#"
                INSERT INTO institutes (code, name, type, state)
                VALUES ($1, $2, $3, 'Unknown')
                ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                "#,
            )
            .bind(inst_code_counter)
            .bind(&record.institute)
            .bind(inst_type)
            .fetch_one(&pool)
            .await?;
            
            inst_code_counter += 1;
            inst_cache.insert(record.institute.clone(), row.0);
            row.0
        };

        // 2. Get or create Program
        let prog_id = if let Some(id) = prog_cache.get(&record.program_name) {
            *id
        } else {
            let p_code = format!("P{}", prog_code_counter);
            let row: (i32,) = sqlx::query_as(
                r#"
                INSERT INTO programs (code, name)
                VALUES ($1, $2)
                ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                "#,
            )
            .bind(&p_code)
            .bind(&record.program_name)
            .fetch_one(&pool)
            .await?;

            prog_code_counter += 1;
            prog_cache.insert(record.program_name.clone(), row.0);
            row.0
        };

        // 3. Parse Seat Type
        // e.g. "OPEN (PwD)", "OBC-NCL", "OPEN"
        let seat_type_upper = record.seat_type.to_uppercase();
        let is_pwd = seat_type_upper.contains("PWD");
        let is_defence = seat_type_upper.contains("DEFENCE") || seat_type_upper.contains("DS");

        let category = if seat_type_upper.starts_with("OPEN") {
            "General"
        } else if seat_type_upper.starts_with("OBC-NCL") {
            "OBC"
        } else if seat_type_upper.starts_with("SC") {
            "SC"
        } else if seat_type_upper.starts_with("ST") {
            "ST"
        } else if seat_type_upper.starts_with("EWS") {
            "EWS"
        } else {
            "General" // fallback
        };

        // Normalise gender: CSV contains verbose values like
        // "Female-only (including Supernumerary)" — collapse to two canonical values.
        let gender = if record.gender.to_lowercase().contains("female") {
            "Female-only"
        } else {
            "Gender-Neutral"
        };

        // Normalise quota: CSV uses "AI", "HS", "OS", "AP", "JK", "LA" etc.
        let quota_norm = match record.quota.as_str() {
            "HS" => "HS",
            _    => "AI",   // AI / OS / AP / JK / LA all treated as All-India
        };

        // Insert into josaa_cutoffs
        sqlx::query(
            r#"
            INSERT INTO josaa_cutoffs (
                institute_id, program_id, year, round, quota, category, gender, is_pwd, is_defence, opening_rank, closing_rank
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(inst_id)
        .bind(prog_id)
        .bind(record.year)
        .bind(record.round)
        .bind(quota_norm)
        .bind(category)
        .bind(gender)
        .bind(is_pwd)
        .bind(is_defence)
        .bind(opening_rank as i32)
        .bind(closing_rank as i32)
        .execute(&pool)
        .await?;

        count += 1;
        if count % 10000 == 0 {
            println!("Inserted {} rows...", count);
        }
    }

    println!("Ingestion complete! Total rows: {}", count);

    Ok(())
}

async fn setup_database(pool: &PgPool) -> Result<(), Box<dyn Error>> {
    // Drop and recreate so re-runs are always clean
    sqlx::query("DROP TABLE IF EXISTS josaa_cutoffs CASCADE").execute(pool).await?;
    sqlx::query("DROP TABLE IF EXISTS programs CASCADE").execute(pool).await?;
    sqlx::query("DROP TABLE IF EXISTS institutes CASCADE").execute(pool).await?;

    sqlx::query(
        r#"
        CREATE TABLE institutes (
            id SERIAL PRIMARY KEY,
            code INT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            type VARCHAR(10) NOT NULL,
            state TEXT NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE programs (
            id SERIAL PRIMARY KEY,
            code VARCHAR(12) UNIQUE NOT NULL,
            name TEXT NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE josaa_cutoffs (
            id BIGSERIAL PRIMARY KEY,
            institute_id INT REFERENCES institutes(id),
            program_id INT REFERENCES programs(id),
            year INT NOT NULL,
            round INT NOT NULL,
            quota VARCHAR(10) NOT NULL,
            category VARCHAR(20) NOT NULL,
            gender VARCHAR(20) NOT NULL,
            is_pwd BOOLEAN DEFAULT FALSE,
            is_defence BOOLEAN DEFAULT FALSE,
            opening_rank INT NOT NULL,
            closing_rank INT NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE INDEX idx_simulation_lookup
        ON josaa_cutoffs (institute_id, program_id, quota, category, gender, is_pwd, is_defence, round);
        "#,
    )
    .execute(pool)
    .await?;

    Ok(())
}
