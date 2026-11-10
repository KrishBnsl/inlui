//! Each quota intersection is an isolated simulatable asset; route by user profile.

use crate::enumerations::enums::{Category, Counselling, IndianState, Quota};

/// Unique key for one legal seat pool, e.g. `DTU_CSE_DelhiRegion_OBC`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuotaAssetKey {
    pub institute: String,
    pub branch: String,
    pub quota: Quota,
    pub category: Category,
    pub counselling: Counselling,
}

/// Horizontal tags layered on vertical category (Female, Defence, PwD, etc.).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HorizontalFlags {
    pub female: bool,
    pub defence: bool,
    pub pwd: bool,
}

/// Student profile used by the routing layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserProfile {
    pub category: Category,
    pub quota: Quota,
    pub homestate: IndianState,
    pub horizontal: HorizontalFlags,
}

/// One row in the quota matrix database.
#[derive(Debug, Clone)]
pub struct QuotaAsset {
    pub key: QuotaAssetKey,
    /// Required for `Quota::HomeState` matching (e.g. Delhi for DTU Delhi region).
    pub institute_home_state: Option<IndianState>,
    /// If set, asset only applies when user carries these horizontal flags.
    pub required_horizontal: HorizontalFlags,
    /// `(year, closing_rank)` for this exact quota slice.
    pub yearly_cutoffs: Vec<(u16, u32)>,
}

#[derive(Debug, Clone, Default)]
pub struct QuotaMatrix {
    pub assets: Vec<QuotaAsset>,
}

impl QuotaMatrix {
    pub fn from_db_rows(rows: Vec<crate::predictor::db::HistoricalCutoff>) -> Self {
        use std::collections::HashMap;

        let mut grouped: HashMap<QuotaAssetKey, QuotaAsset> = HashMap::new();

        for row in rows {
            let quota_val = match row.quota.as_str() {
                "HomeState" | "HS" => Quota::HomeState,
                "OtherState" | "OS" | "AI" | "AIQ" => Quota::OtherState,
                _ => Quota::OtherState,
            };

            let category_val = match row.category.as_str() {
                "General" | "OPEN" => Category::General,
                "OBC" | "OBC-NCL" => Category::OBC,
                "SC" => Category::SC,
                "ST" => Category::ST,
                "EWS" => Category::EWS,
                "GirlChild" => Category::GirlChild,
                "KashmiriMigrant" => Category::KashmiriMigrant,
                "PwD" => Category::PwD,
                "Defence" | "DS" => Category::Defence,
                _ => Category::General,
            };

            let key = QuotaAssetKey {
                institute: row.institute_name.clone(),
                branch: row.program_name.clone(),
                quota: quota_val,
                category: category_val,
                counselling: Counselling::JoSAA,
            };

            let asset = grouped.entry(key.clone()).or_insert_with(|| QuotaAsset {
                key,
                institute_home_state: parse_state(&row.state),
                required_horizontal: HorizontalFlags {
                    female: row.gender.contains("Female") || row.gender.contains("female"),
                    defence: row.is_defence,
                    pwd: row.is_pwd,
                },
                yearly_cutoffs: Vec::new(),
            });

            asset
                .yearly_cutoffs
                .push((row.year as u16, row.closing_rank as u32));
        }

        Self {
            assets: grouped.into_values().collect(),
        }
    }

    pub fn route<'a>(&'a self, user: &UserProfile) -> Vec<&'a QuotaAsset> {
        self.assets
            .iter()
            .filter(|asset| is_eligible(user, asset))
            .collect()
    }

    pub fn find_by_key(&self, key: &QuotaAssetKey) -> Option<&QuotaAsset> {
        self.assets.iter().find(|a| &a.key == key)
    }
}

fn parse_state(state_str: &str) -> Option<IndianState> {
    use IndianState::*;
    match state_str {
        "Andhra Pradesh" => Some(AndhraPradesh),
        "Arunachal Pradesh" => Some(ArunachalPradesh),
        "Assam" => Some(Assam),
        "Bihar" => Some(Bihar),
        "Chhattisgarh" => Some(Chhattisgarh),
        "Goa" => Some(Goa),
        "Gujarat" => Some(Gujarat),
        "Haryana" => Some(Haryana),
        "Himachal Pradesh" => Some(HimachalPradesh),
        "Jharkhand" => Some(Jharkhand),
        "Karnataka" => Some(Karnataka),
        "Kerala" => Some(Kerala),
        "Madhya Pradesh" => Some(MadhyaPradesh),
        "Maharashtra" => Some(Maharashtra),
        "Manipur" => Some(Manipur),
        "Meghalaya" => Some(Meghalaya),
        "Mizoram" => Some(Mizoram),
        "Nagaland" => Some(Nagaland),
        "Odisha" => Some(Odisha),
        "Punjab" => Some(Punjab),
        "Rajasthan" => Some(Rajasthan),
        "Sikkim" => Some(Sikkim),
        "Tamil Nadu" => Some(TamilNadu),
        "Telangana" => Some(Telangana),
        "Tripura" => Some(Tripura),
        "Uttar Pradesh" => Some(UttarPradesh),
        "Uttarakhand" => Some(Uttarakhand),
        "West Bengal" => Some(WestBengal),
        "Delhi" => Some(Delhi),
        "Chandigarh" => Some(Chandigarh),
        _ => None,
    }
}

/// Legal eligibility gate — only assets the user can actually claim.
pub fn is_eligible(user: &UserProfile, asset: &QuotaAsset) -> bool {
    if user.category != asset.key.category {
        return false;
    }
    if user.quota != asset.key.quota {
        return false;
    }
    if !horizontal_satisfied(user.horizontal, asset.required_horizontal) {
        return false;
    }
    match asset.key.quota {
        Quota::HomeState => {
            let Some(home) = asset.institute_home_state else {
                return false;
            };
            user.homestate == home
        }
        Quota::OtherState => true,
    }
}

fn horizontal_satisfied(user: HorizontalFlags, required: HorizontalFlags) -> bool {
    (!required.female || user.female)
        && (!required.defence || user.defence)
        && (!required.pwd || user.pwd)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dtu_obc_delhi_asset() -> QuotaAsset {
        QuotaAsset {
            key: QuotaAssetKey {
                institute: "DTU".into(),
                branch: "CSE".into(),
                quota: Quota::HomeState,
                category: Category::OBC,
                counselling: Counselling::JacDelhi,
            },
            institute_home_state: Some(IndianState::Delhi),
            required_horizontal: HorizontalFlags::default(),
            yearly_cutoffs: vec![(2024, 12_000)],
        }
    }

    #[test]
    fn routes_only_matching_quota_slice() {
        let matrix = QuotaMatrix {
            assets: vec![dtu_obc_delhi_asset()],
        };
        let delhi_obc = UserProfile {
            category: Category::OBC,
            quota: Quota::HomeState,
            homestate: IndianState::Delhi,
            horizontal: HorizontalFlags::default(),
        };
        let punjab_obc = UserProfile {
            homestate: IndianState::Punjab,
            ..delhi_obc
        };
        assert_eq!(matrix.route(&delhi_obc).len(), 1);
        assert_eq!(matrix.route(&punjab_obc).len(), 0);
    }
}
