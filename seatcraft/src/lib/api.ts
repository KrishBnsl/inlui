import type {
  SimulationInput,
  SimulationResponse,
  PredictionResult,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export async function runSimulation(
  input: SimulationInput
): Promise<SimulationResponse> {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(`Simulation failed: ${await res.text()}`);
  return res.json();
}

// ─── Realistic JoSAA Seat Nodes ───────────────────────────────────────────────
// Closing ranks are Round 6 actuals. OPEN / Gender-Neutral.
// AI  = All India quota cutoffs.
// HS  = Home State quota cutoffs (null means institute has no HS seats, e.g. IITs).
// IIT rank scale: ~1–15,000 (JEE Advanced)
// NIT/IIIT/GFTI rank scale: ~1–12,00,000 (JEE Main)
type HistoricalSlice = [number, number, number, number]; // [2022,2023,2024,2025]

type SeatNode = {
  institute_name: string;
  institute_type: "IIT" | "NIT" | "IIIT" | "GFTI";
  program_name: string;
  rank_type: "advanced" | "main";
  institute_state: string | null; // null for IITs (AI only, no HS)
  ai_closing: HistoricalSlice;
  ai_opening: HistoricalSlice;
  // HS quota: null means this institute has no HS quota (IITs, some GFTIs)
  hs_closing: HistoricalSlice | null;
  hs_opening: HistoricalSlice | null;
};

const SEAT_NODES: SeatNode[] = [
  // ── IITs (JEE Advanced, AI only — no HS quota) ────────────────────────────
  {
    institute_name: "Indian Institute of Technology Bombay",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Computer Science and Engineering",
    ai_closing: [63, 67, 61, 65],
    ai_opening: [1, 1, 1, 1],
    hs_closing: null,
    hs_opening: null,
  },
  {
    institute_name: "Indian Institute of Technology Bombay",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Electrical Engineering",
    ai_closing: [498, 512, 487, 505],
    ai_opening: [220, 235, 215, 228],
    hs_closing: null,
    hs_opening: null,
  },
  {
    institute_name: "Indian Institute of Technology Delhi",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Computer Science and Engineering",
    ai_closing: [118, 125, 112, 120],
    ai_opening: [55, 60, 52, 58],
    hs_closing: null,
    hs_opening: null,
  },
  {
    institute_name: "Indian Institute of Technology Delhi",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Electrical Engineering",
    ai_closing: [675, 692, 660, 680],
    ai_opening: [340, 355, 330, 348],
    hs_closing: null,
    hs_opening: null,
  },
  {
    institute_name: "Indian Institute of Technology Madras",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Computer Science and Engineering",
    ai_closing: [103, 110, 98, 106],
    ai_opening: [42, 45, 40, 44],
    hs_closing: null,
    hs_opening: null,
  },
  {
    institute_name: "Indian Institute of Technology Kanpur",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Computer Science and Engineering",
    ai_closing: [286, 298, 274, 290],
    ai_opening: [120, 130, 115, 126],
    hs_closing: null,
    hs_opening: null,
  },
  {
    institute_name: "Indian Institute of Technology Kharagpur",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Computer Science and Engineering",
    ai_closing: [452, 468, 438, 458],
    ai_opening: [200, 215, 192, 208],
    hs_closing: null,
    hs_opening: null,
  },
  {
    institute_name: "Indian Institute of Technology Roorkee",
    institute_type: "IIT",
    rank_type: "advanced",
    institute_state: null,
    program_name: "Computer Science and Engineering",
    ai_closing: [1048, 1075, 1020, 1056],
    ai_opening: [520, 540, 505, 530],
    hs_closing: null,
    hs_opening: null,
  },

  // ── NITs (JEE Main, both AI and HS quotas) ────────────────────────────────
  // NIT Trichy — Tamil Nadu
  {
    institute_name: "National Institute of Technology Trichy",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Tamil Nadu",
    program_name: "Computer Science and Engineering",
    ai_closing: [1248, 1302, 1195, 1265],
    ai_opening: [620, 650, 598, 635],
    // HS: Tamil Nadu students compete in a smaller pool; cutoff is higher (worse rank)
    hs_closing: [4820, 5050, 4650, 4890],
    hs_opening: [2340, 2460, 2270, 2380],
  },
  {
    institute_name: "National Institute of Technology Trichy",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Tamil Nadu",
    program_name: "Electronics and Communication Engineering",
    ai_closing: [3580, 3720, 3450, 3610],
    ai_opening: [1820, 1895, 1760, 1845],
    hs_closing: [10200, 10650, 9870, 10320],
    hs_opening: [5100, 5320, 4940, 5160],
  },
  // NIT Surathkal — Karnataka
  {
    institute_name: "National Institute of Technology Surathkal",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Karnataka",
    program_name: "Computer Science and Engineering",
    ai_closing: [2840, 2960, 2740, 2880],
    ai_opening: [1380, 1440, 1330, 1400],
    hs_closing: [5640, 5890, 5450, 5720],
    hs_opening: [2760, 2890, 2670, 2800],
  },
  // NIT Warangal — Telangana
  {
    institute_name: "National Institute of Technology Warangal",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Telangana",
    program_name: "Computer Science and Engineering",
    ai_closing: [4150, 4320, 4020, 4200],
    ai_opening: [2050, 2140, 1985, 2080],
    hs_closing: [8920, 9280, 8620, 9010],
    hs_opening: [4380, 4560, 4230, 4430],
  },
  // NIT Calicut — Kerala
  {
    institute_name: "National Institute of Technology Calicut",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Kerala",
    program_name: "Computer Science and Engineering",
    ai_closing: [5620, 5840, 5440, 5690],
    ai_opening: [2780, 2890, 2690, 2810],
    hs_closing: [7840, 8150, 7580, 7920],
    hs_opening: [3840, 4000, 3720, 3880],
  },
  // NIT Rourkela — Odisha
  {
    institute_name: "National Institute of Technology Rourkela",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Odisha",
    program_name: "Computer Science and Engineering",
    ai_closing: [6890, 7180, 6650, 6980],
    ai_opening: [3360, 3500, 3250, 3410],
    // Odisha has fewer JEE qualifiers → HS pool is actually relatively easier
    hs_closing: [5820, 6090, 5640, 5920],
    hs_opening: [2840, 2980, 2760, 2890],
  },
  // MNIT Jaipur — Rajasthan
  {
    institute_name: "Malaviya National Institute of Technology Jaipur",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Rajasthan",
    program_name: "Computer Science and Engineering",
    ai_closing: [8420, 8760, 8140, 8520],
    ai_opening: [4120, 4290, 3990, 4190],
    hs_closing: [12840, 13360, 12410, 12980],
    hs_opening: [6240, 6510, 6050, 6320],
  },
  // NIT Kurukshetra — Haryana
  {
    institute_name: "National Institute of Technology Kurukshetra",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Haryana",
    program_name: "Computer Science and Engineering",
    ai_closing: [10240, 10680, 9880, 10360],
    ai_opening: [5050, 5280, 4880, 5120],
    hs_closing: [15620, 16280, 15080, 15840],
    hs_opening: [7620, 7960, 7380, 7720],
  },

  // ── IIITs (JEE Main, both AI and HS quotas) ───────────────────────────────
  // IIIT Allahabad — Uttar Pradesh
  {
    institute_name: "Indian Institute of Information Technology Allahabad",
    institute_type: "IIIT",
    rank_type: "main",
    institute_state: "Uttar Pradesh",
    program_name: "Computer Science and Engineering",
    ai_closing: [3920, 4080, 3780, 3970],
    ai_opening: [1920, 2000, 1860, 1950],
    hs_closing: [6840, 7120, 6610, 6920],
    hs_opening: [3340, 3490, 3240, 3390],
  },
  // IIIT Hyderabad — Telangana
  {
    institute_name: "Indian Institute of Information Technology Hyderabad",
    institute_type: "IIIT",
    rank_type: "main",
    institute_state: "Telangana",
    program_name: "Computer Science and Engineering",
    ai_closing: [5140, 5360, 4960, 5200],
    ai_opening: [2520, 2630, 2440, 2570],
    hs_closing: [9280, 9680, 8970, 9420],
    hs_opening: [4540, 4740, 4390, 4620],
  },
  // IIIT DM Kancheepuram — Tamil Nadu
  {
    institute_name: "Indian Institute of Information Technology DM Kancheepuram",
    institute_type: "IIIT",
    rank_type: "main",
    institute_state: "Tamil Nadu",
    program_name: "Computer Science and Engineering",
    ai_closing: [9860, 10280, 9530, 9980],
    ai_opening: [4810, 5020, 4660, 4890],
    hs_closing: [18420, 19180, 17840, 18680],
    hs_opening: [8980, 9360, 8710, 9120],
  },

  // ── GFTIs (JEE Main, HS only for state-specific ones) ────────────────────
  // DTU — Delhi (Delhi-domicile HS quota exists but is very limited)
  {
    institute_name: "Delhi Technological University",
    institute_type: "GFTI",
    rank_type: "main",
    institute_state: "Delhi",
    program_name: "Computer Science and Engineering",
    ai_closing: [4820, 5030, 4660, 4890],
    ai_opening: [2350, 2460, 2280, 2390],
    hs_closing: [3640, 3810, 3530, 3720],
    hs_opening: [1780, 1870, 1730, 1820],
  },
  // NSUT — Delhi
  {
    institute_name: "Netaji Subhas University of Technology",
    institute_type: "GFTI",
    rank_type: "main",
    institute_state: "Delhi",
    program_name: "Computer Science and Engineering",
    ai_closing: [6320, 6580, 6110, 6400],
    ai_opening: [3090, 3220, 2990, 3140],
    hs_closing: [4920, 5140, 4760, 4990],
    hs_opening: [2400, 2510, 2330, 2440],
  },

  // ── More NITs — mid-tier cutoffs (15k–40k AI) ─────────────────────────────
  {
    institute_name: "National Institute of Technology Patna",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Bihar",
    program_name: "Computer Science and Engineering",
    ai_closing: [14820, 15460, 14310, 15080],
    ai_opening: [7240, 7560, 7010, 7380],
    hs_closing: [12640, 13180, 12200, 12880],
    hs_opening: [6180, 6450, 5980, 6280],
  },
  {
    institute_name: "National Institute of Technology Silchar",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Assam",
    program_name: "Computer Science and Engineering",
    ai_closing: [18640, 19420, 18020, 18910],
    ai_opening: [9120, 9510, 8830, 9260],
    hs_closing: [8420, 8780, 8140, 8540],
    hs_opening: [4120, 4300, 3990, 4190],
  },
  {
    institute_name: "National Institute of Technology Durgapur",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "West Bengal",
    program_name: "Computer Science and Engineering",
    ai_closing: [22480, 23410, 21710, 22840],
    ai_opening: [10980, 11450, 10610, 11160],
    hs_closing: [19640, 20480, 18990, 19920],
    hs_opening: [9600, 10010, 9290, 9760],
  },
  {
    institute_name: "Visvesvaraya National Institute of Technology Nagpur",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Maharashtra",
    program_name: "Computer Science and Engineering",
    ai_closing: [16940, 17660, 16370, 17180],
    ai_opening: [8280, 8640, 8010, 8400],
    hs_closing: [21840, 22760, 21120, 22180],
    hs_opening: [10680, 11130, 10330, 10840],
  },
  {
    institute_name: "Maulana Azad National Institute of Technology Bhopal",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Madhya Pradesh",
    program_name: "Computer Science and Engineering",
    ai_closing: [25640, 26720, 24780, 25980],
    ai_opening: [12540, 13080, 12120, 12740],
    hs_closing: [23180, 24140, 22390, 23520],
    hs_opening: [11330, 11810, 10950, 11490],
  },
  {
    institute_name: "National Institute of Technology Hamirpur",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Himachal Pradesh",
    program_name: "Computer Science and Engineering",
    ai_closing: [32840, 34220, 31760, 33280],
    ai_opening: [16060, 16750, 15540, 16280],
    hs_closing: [11240, 11720, 10880, 11380],
    hs_opening: [5500, 5740, 5330, 5570],
  },
  {
    institute_name: "National Institute of Technology Srinagar",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Jammu and Kashmir",
    program_name: "Computer Science and Engineering",
    ai_closing: [42180, 43960, 40780, 42640],
    ai_opening: [20620, 21510, 19940, 20850],
    hs_closing: [6840, 7140, 6620, 6940],
    hs_opening: [3350, 3500, 3250, 3400],
  },
  {
    institute_name: "National Institute of Technology Agartala",
    institute_type: "NIT",
    rank_type: "main",
    institute_state: "Tripura",
    program_name: "Computer Science and Engineering",
    ai_closing: [58420, 60880, 56490, 59140],
    ai_opening: [28560, 29780, 27620, 28940],
    hs_closing: [4820, 5040, 4670, 4890],
    hs_opening: [2360, 2470, 2290, 2390],
  },

  // ── More IIITs — mid-tier cutoffs (12k–60k AI) ────────────────────────────
  {
    institute_name: "Indian Institute of Information Technology Gwalior",
    institute_type: "IIIT",
    rank_type: "main",
    institute_state: "Madhya Pradesh",
    program_name: "Computer Science and Engineering",
    ai_closing: [12840, 13380, 12420, 12980],
    ai_opening: [6280, 6550, 6080, 6340],
    hs_closing: [22480, 23420, 21730, 22760],
    hs_opening: [10980, 11460, 10630, 11130],
  },
  {
    institute_name: "Indian Institute of Information Technology Vadodara",
    institute_type: "IIIT",
    rank_type: "main",
    institute_state: "Gujarat",
    program_name: "Computer Science and Engineering",
    ai_closing: [21640, 22580, 20930, 21940],
    ai_opening: [10580, 11040, 10240, 10740],
    hs_closing: [28420, 29620, 27490, 28780],
    hs_opening: [13900, 14500, 13460, 14080],
  },
  {
    institute_name: "Indian Institute of Information Technology Sri City",
    institute_type: "IIIT",
    rank_type: "main",
    institute_state: "Andhra Pradesh",
    program_name: "Computer Science and Engineering",
    ai_closing: [34820, 36310, 33680, 35240],
    ai_opening: [17040, 17770, 16490, 17240],
    hs_closing: [52640, 54880, 50940, 53280],
    hs_opening: [25740, 26840, 24920, 26000],
  },
  {
    institute_name: "Indian Institute of Information Technology Una",
    institute_type: "IIIT",
    rank_type: "main",
    institute_state: "Himachal Pradesh",
    program_name: "Computer Science and Engineering",
    ai_closing: [48640, 50720, 47060, 49240],
    ai_opening: [23800, 24820, 23030, 24100],
    hs_closing: [16840, 17560, 16290, 17080],
    hs_opening: [8240, 8600, 7980, 8350],
  },

  // ── GFTIs — wide cutoff range (8k–150k AI) ────────────────────────────────
  {
    institute_name: "Indian Institute of Engineering Science and Technology Shibpur",
    institute_type: "GFTI",
    rank_type: "main",
    institute_state: "West Bengal",
    program_name: "Computer Science and Engineering",
    ai_closing: [13840, 14440, 13390, 14020],
    ai_opening: [6780, 7080, 6560, 6860],
    hs_closing: [10640, 11100, 10290, 10810],
    hs_opening: [5210, 5440, 5050, 5300],
  },
  {
    institute_name: "Assam University Silchar",
    institute_type: "GFTI",
    rank_type: "main",
    institute_state: "Assam",
    program_name: "Computer Science and Engineering",
    ai_closing: [78640, 81980, 76060, 79540],
    ai_opening: [38480, 40140, 37240, 38760],
    hs_closing: [24840, 25920, 24040, 25140],
    hs_opening: [12160, 12690, 11780, 12300],
  },
  {
    institute_name: "Gurukula Kangri Vishwavidyalaya Haridwar",
    institute_type: "GFTI",
    rank_type: "main",
    institute_state: "Uttarakhand",
    program_name: "Computer Science and Engineering",
    ai_closing: [112840, 117680, 109120, 114160],
    ai_opening: [55240, 57620, 53400, 55780],
    hs_closing: [68420, 71360, 66200, 69200],
    hs_opening: [33480, 34940, 32400, 33840],
  },
  {
    institute_name: "National Institute of Foundry and Forge Technology Hatia",
    institute_type: "GFTI",
    rank_type: "main",
    institute_state: "Jharkhand",
    program_name: "Computer Science and Engineering",
    ai_closing: [164820, 171820, 159440, 166720],
    ai_opening: [80680, 84100, 78040, 81440],
    hs_closing: [42840, 44680, 41440, 43360],
    hs_opening: [20960, 21870, 20280, 21220],
  },
];


// ─── Quota routing ────────────────────────────────────────────────────────────
// If student's home_state matches institute_state → use HS quota data.
// IITs always use AI (institute_state is null).
function resolveQuota(
  homeState: string,
  node: SeatNode
): { quota: "AI" | "HS"; closing: HistoricalSlice; opening: HistoricalSlice } {
  if (
    node.institute_state !== null &&
    node.hs_closing !== null &&
    node.hs_opening !== null &&
    homeState === node.institute_state
  ) {
    return { quota: "HS", closing: node.hs_closing, opening: node.hs_opening };
  }
  return { quota: "AI", closing: node.ai_closing, opening: node.ai_opening };
}

// ─── Probability engine ───────────────────────────────────────────────────────
// Gaussian CDF: P(user_rank ≤ simulated_cutoff) where cutoff ~ N(μ, σ)
function computeProbability(
  userRank: number,
  closing: HistoricalSlice
): number {
  const n = closing.length;
  const mean = closing.reduce((s, v) => s + v, 0) / n;
  const variance = closing.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
  const stdDev = Math.sqrt(variance);

  if (stdDev <= 0) return userRank <= mean ? 99.5 : 0.5;

  const z = (mean - userRank) / stdDev;
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const poly =
    t *
    (0.31938153 +
      t *
        (-0.356563782 +
          t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const cdf =
    1 - (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * z * z) * poly;
  return Math.max(0.5, Math.min(99.5, (z >= 0 ? cdf : 1 - cdf) * 100));
}

// ─── Mock (swap → runSimulation when backend is live) ─────────────────────────
export async function runSimulationMock(
  input: SimulationInput
): Promise<SimulationResponse> {
  await new Promise((r) => setTimeout(r, 2200));

  // Only include IIT nodes if student has an Advanced rank
  const eligibleNodes = SEAT_NODES.filter(
    (node) => node.rank_type === "main" || input.advanced_rank !== undefined
  );

  const results: PredictionResult[] = eligibleNodes.map((node, i) => {
    const userRank =
      node.rank_type === "advanced"
        ? (input.advanced_rank ?? Infinity)
        : input.main_rank;

    const { quota, closing, opening } = resolveQuota(input.home_state, node);

    const prob = computeProbability(userRank, closing);
    const meanClosing = closing.reduce((s, v) => s + v, 0) / closing.length;

    return {
      id: String(i),
      institute_name: node.institute_name,
      institute_type: node.institute_type,
      program_name: node.program_name,
      quota_applied: quota,
      category: input.category,
      probability_percent: Math.round(prob * 10) / 10,
      projected_closing_rank: Math.round(meanClosing),
      historical_data: ([2022, 2023, 2024, 2025] as const).map(
        (year, idx) => ({
          year,
          opening_rank: opening[idx],
          closing_rank: closing[idx],
        })
      ),
    };
  });

  const sorted = results.sort(
    (a, b) => b.probability_percent - a.probability_percent
  );

  return {
    total_options: sorted.length,
    safest_choice: sorted[0]?.institute_name ?? "—",
    top_upgrade: sorted[Math.floor(sorted.length * 0.33)]?.institute_name ?? "—",
    results: sorted,
  };
}
