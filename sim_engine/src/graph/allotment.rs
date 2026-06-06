//! Student-proposing Gale–Shapley allotment.
//!
//! Students propose to seats in preference order. Each seat keeps up to
//! `max_seat_capacity` students, preferring lower `rank` (better exam rank).
//! When a seat is full, the worst-ranked occupant is bumped and may propose again.

use std::collections::HashMap;

use crate::enumerations::enums::counselling;
use crate::graph::def::{AdmissionGraph, CandidateId, CandidateNode, NodeId, SeatNode};

/// Result of a single Gale–Shapley run.
#[derive(Debug, Clone, Default)]
pub struct AllotmentResult {
    /// Final seat assignment per candidate (`None` if unplaced).
    pub assignments: HashMap<CandidateId, Option<NodeId>>,
    /// Candidates who exhausted their preference list without a seat.
    pub unallocated: Vec<CandidateId>,
}

/// Options for filtering which seats participate in a round.
#[derive(Debug, Clone, Default)]
pub struct AllotmentOptions {
    /// When empty, seats from any counselling are considered.
    /// Otherwise only seats whose `counselling` is in this list are used.
    pub counsellings: Vec<counselling>,
}

impl AdmissionGraph {
    /// Run Gale–Shapley and write outcomes into each candidate's `allocated_seat`.
    pub fn run_gale_shapley(&mut self, options: AllotmentOptions) -> AllotmentResult {
        let result = gale_shapley(self, options);
        apply_assignments_to_candidates(self, &result);
        result
    }
}

/// Core Gale–Shapley (does not mutate `allocated_seat` on candidates).
pub fn gale_shapley(graph: &AdmissionGraph, options: AllotmentOptions) -> AllotmentResult {
    let expanded = expand_preferences(graph, options);
    let mut next_proposal: HashMap<CandidateId, usize> =
        expanded.keys().copied().map(|id| (id, 0)).collect();

    let mut free: Vec<CandidateId> = expanded.keys().copied().collect();
    // Process better-ranked students first when multiple propose in the same pass.
    free.sort_by_key(|id| graph.candidate_nodes[id].rank);

    let mut occupants: HashMap<NodeId, Vec<CandidateId>> = HashMap::new();
    let mut assignments: HashMap<CandidateId, Option<NodeId>> = expanded
        .keys()
        .copied()
        .map(|id| (id, None))
        .collect();

    while let Some(candidate_id) = free.pop() {
        let pref_list = match expanded.get(&candidate_id) {
            Some(list) => list,
            None => continue,
        };

        let pref_idx = next_proposal[&candidate_id];
        if pref_idx >= pref_list.len() {
            assignments.insert(candidate_id, None);
            continue;
        }

        let seat_id = pref_list[pref_idx];
        next_proposal.insert(candidate_id, pref_idx + 1);

        let Some(seat) = graph.seat_nodes.get(&seat_id) else {
            free.push(candidate_id);
            continue;
        };

        // A student holds at most one tentative seat; clear any previous one before proposing.
        if let Some(Some(previous_seat)) = assignments.get(&candidate_id).copied() {
            if let Some(previous_holder) = occupants.get_mut(&previous_seat) {
                previous_holder.retain(|&id| id != candidate_id);
            }
        }

        let holder = occupants.entry(seat_id).or_default();
        holder.push(candidate_id);
        holder.sort_by_key(|id| graph.candidate_nodes[id].rank);

        let capacity = seat.max_seat_capacity as usize;
        let rejected: Vec<CandidateId> = holder.drain(capacity..).collect();

        if holder.contains(&candidate_id) {
            assignments.insert(candidate_id, Some(seat_id));
            for bumped in &rejected {
                if assignments.get(bumped) == Some(&Some(seat_id)) {
                    assignments.insert(*bumped, None);
                }
                free.push(*bumped);
            }
        } else {
            assignments.insert(candidate_id, None);
            free.push(candidate_id);
            for bumped in rejected {
                if assignments.get(&bumped) == Some(&Some(seat_id)) {
                    assignments.insert(bumped, None);
                }
                free.push(bumped);
            }
        }
    }

    let unallocated: Vec<CandidateId> = assignments
        .iter()
        .filter_map(|(id, seat)| seat.is_none().then_some(*id))
        .collect();

    AllotmentResult {
        assignments,
        unallocated,
    }
}

fn apply_assignments_to_candidates(graph: &mut AdmissionGraph, result: &AllotmentResult) {
    for (candidate_id, seat_id) in &result.assignments {
        let allocated = seat_id.and_then(|sid| {
            graph.seat_nodes.get(&sid).map(|seat| {
                (
                    seat.insti.clone(),
                    seat.branch.preference_label().to_string(),
                )
            })
        });
        if let Some(candidate) = graph.candidate_nodes.get_mut(candidate_id) {
            candidate.allocated_seat = allocated;
        }
    }
}

/// Expand each `(institute, branch)` preference into concrete, eligible seat ids.
fn expand_preferences(
    graph: &AdmissionGraph,
    options: AllotmentOptions,
) -> HashMap<CandidateId, Vec<NodeId>> {
    let mut out = HashMap::new();

    for (candidate_id, candidate) in &graph.candidate_nodes {
        let mut seats = Vec::new();
        for (insti, branch) in &candidate.preferences {
            let mut matches: Vec<NodeId> = graph
                .seat_nodes
                .iter()
                .filter(|(_, seat)| {
                    seat.insti == *insti
                        && seat.branch.matches_preference(branch)
                        && is_eligible(candidate, seat)
                        && counselling_allowed(seat, &options.counsellings)
                })
                .map(|(id, _)| *id)
                .collect();
            matches.sort_unstable();
            seats.extend(matches);
        }
        out.insert(*candidate_id, seats);
    }

    out
}

fn is_eligible(candidate: &CandidateNode, seat: &SeatNode) -> bool {
    candidate.category == seat.category
}

fn counselling_allowed(seat: &SeatNode, allowed: &[counselling]) -> bool {
    allowed.is_empty() || allowed.contains(&seat.counselling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerations::enums::{
        category, counselling, quota, CseSpecialization, EngineeringBranch,
    };
    use crate::graph::def::{CandidateNode, SeatNode};

    fn seat(
        id: NodeId,
        insti: &str,
        branch: EngineeringBranch,
        capacity: u16,
        cat: category,
    ) -> SeatNode {
        SeatNode {
            id,
            insti: insti.into(),
            branch,
            quota: quota::OtherState,
            category: cat,
            max_seat_capacity: capacity,
            pred_closing_rank: 0,
            counselling: counselling::JoSAA,
        }
    }

    fn candidate(
        id: CandidateId,
        rank: u32,
        prefs: Vec<(&str, &str)>,
    ) -> CandidateNode {
        CandidateNode {
            id,
            name: format!("c{id}"),
            rank,
            category: category::General,
            homestate: crate::enumerations::enums::IndianState::Delhi,
            preferences: prefs
                .into_iter()
                .map(|(i, b)| (i.to_string(), b.to_string()))
                .collect(),
            allocated_seat: None,
        }
    }

    #[test]
    fn higher_rank_student_gets_seat_when_capacity_is_one() {
        let mut graph = AdmissionGraph {
            seat_nodes: HashMap::from([(
                1,
                seat(
                    1,
                    "IIT-A",
                    EngineeringBranch::ComputerScience(CseSpecialization::Core),
                    1,
                    category::General,
                ),
            )]),
            candidate_nodes: HashMap::from([
                (1, candidate(1, 100, vec![("IIT-A", "CSE")])),
                (2, candidate(2, 50, vec![("IIT-A", "CSE")])),
            ]),
            edges: HashMap::new(),
        };

        let result = graph.run_gale_shapley(AllotmentOptions::default());
        assert_eq!(result.assignments.get(&2), Some(&Some(1)));
        assert_eq!(result.assignments.get(&1), Some(&None));
    }

    #[test]
    fn student_gets_second_preference_when_first_is_full() {
        let mut graph = AdmissionGraph {
            seat_nodes: HashMap::from([
                (
                    1,
                    seat(
                        1,
                        "IIT-A",
                        EngineeringBranch::ComputerScience(CseSpecialization::Core),
                        1,
                        category::General,
                    ),
                ),
                (
                    2,
                    seat(
                        2,
                        "IIT-B",
                        EngineeringBranch::ComputerScience(CseSpecialization::Core),
                        1,
                        category::General,
                    ),
                ),
            ]),
            candidate_nodes: HashMap::from([
                (1, candidate(1, 20, vec![("IIT-A", "CSE"), ("IIT-B", "CSE")])),
                (2, candidate(2, 10, vec![("IIT-A", "CSE")])),
            ]),
            edges: HashMap::new(),
        };

        let result = graph.run_gale_shapley(AllotmentOptions::default());
        // Better rank (10) takes IIT-A; the other student falls through to IIT-B.
        assert_eq!(result.assignments.get(&2), Some(&Some(1)));
        assert_eq!(result.assignments.get(&1), Some(&Some(2)));
    }
}
