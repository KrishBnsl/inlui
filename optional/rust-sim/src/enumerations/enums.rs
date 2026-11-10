#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quota {
    HomeState,
    OtherState, //also called All India Quota (AIQ)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    General,
    OBC,
    SC,
    ST,
    EWS,
    GirlChild,
    KashmiriMigrant,
    PwD,
    Defence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Counselling {
    JoSAA,
    JacDelhi,
    JacChandigarh,
    BITS,
    VITEEE,
    IPU,
    Manipal,
    UGEE,
}

/// Sub-specializations typically offered under Computer Science and Engineering
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CseSpecialization {
    Core,
    ArtificialIntelligenceAndMachineLearning,
    DataScienceAndAnalytics,
    CyberSecurityAndForensics,
    CloudComputingAndDevOps,
    InternetOfThings,
    BlockchainTechnology,
    DataScienceAndArtificialIntelligence,
}

/// A comprehensive representation of engineering branches available in India
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EngineeringBranch {
    // Computing & Information Technology
    ComputerScience(CseSpecialization),
    InformationTechnology,
    SoftwareEngineering,

    // Core Engineering Branches
    Mechanical,
    Civil,
    Electrical,
    Chemical,

    // Electronics & Communication
    ElectronicsAndCommunication, // ECE
    ElectricalAndElectronics,    // EEE
    InstrumentationAndControl,

    // Specialized & Advanced Engineering
    Aerospace, // Also covers Aeronautical
    Automobile,
    Biotechnology,
    Biomedical,
    Mechatronics,
    Robotics,

    // Industrial, Materials & Earth Sciences
    ProductionAndIndustrial,
    MetallurgicalAndMaterials,
    Mining,
    Petroleum,
    Environmental,

    // Niche & Emerging Branches
    Agricultural,
    FoodTechnology,
    Marine,
    Textile,
    Nuclear,
}

impl EngineeringBranch {
    /// Helper method to check if the branch is related to the computing domain
    pub fn is_computing_branch(&self) -> bool {
        matches!(
            self,
            Self::ComputerScience(_) | Self::InformationTechnology | Self::SoftwareEngineering
        )
    }

    /// Label used in candidate preference tuples `(institute, branch)`.
    pub fn preference_label(&self) -> &'static str {
        match self {
            Self::ComputerScience(CseSpecialization::Core) => "CSE",
            Self::ComputerScience(CseSpecialization::ArtificialIntelligenceAndMachineLearning) => {
                "CSE-AIML"
            }
            Self::ComputerScience(CseSpecialization::DataScienceAndAnalytics) => "CSE-DS",
            Self::ComputerScience(CseSpecialization::CyberSecurityAndForensics) => "CSE-Cyber",
            Self::ComputerScience(CseSpecialization::CloudComputingAndDevOps) => "CSE-Cloud",
            Self::ComputerScience(CseSpecialization::InternetOfThings) => "CSE-IoT",
            Self::ComputerScience(CseSpecialization::BlockchainTechnology) => "CSE-Blockchain",
            Self::ComputerScience(CseSpecialization::DataScienceAndArtificialIntelligence) => {
                "CSE-DSAI"
            }
            Self::InformationTechnology => "IT",
            Self::SoftwareEngineering => "Software",
            Self::Mechanical => "Mechanical",
            Self::Civil => "Civil",
            Self::Electrical => "Electrical",
            Self::Chemical => "Chemical",
            Self::ElectronicsAndCommunication => "ECE",
            Self::ElectricalAndElectronics => "EEE",
            Self::InstrumentationAndControl => "ICE",
            Self::Aerospace => "Aerospace",
            Self::Automobile => "Automobile",
            Self::Biotechnology => "Biotech",
            Self::Biomedical => "Biomedical",
            Self::Mechatronics => "Mechatronics",
            Self::Robotics => "Robotics",
            Self::ProductionAndIndustrial => "Production",
            Self::MetallurgicalAndMaterials => "Metallurgy",
            Self::Mining => "Mining",
            Self::Petroleum => "Petroleum",
            Self::Environmental => "Environmental",
            Self::Agricultural => "Agricultural",
            Self::FoodTechnology => "Food",
            Self::Marine => "Marine",
            Self::Textile => "Textile",
            Self::Nuclear => "Nuclear",
        }
    }

    pub fn matches_preference(&self, branch_label: &str) -> bool {
        self.preference_label().eq_ignore_ascii_case(branch_label)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndianState {
    AndhraPradesh,
    ArunachalPradesh,
    Assam,
    Bihar,
    Chhattisgarh,
    Goa,
    Gujarat,
    Haryana,
    HimachalPradesh,
    Jharkhand,
    Karnataka,
    Kerala,
    MadhyaPradesh,
    Maharashtra,
    Manipur,
    Meghalaya,
    Mizoram,
    Nagaland,
    Odisha,
    Punjab,
    Rajasthan,
    Sikkim,
    TamilNadu,
    Telangana,
    Tripura,
    UttarPradesh,
    Uttarakhand,
    WestBengal,
    Delhi,
    Chandigarh,
}
