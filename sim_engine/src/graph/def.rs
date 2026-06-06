use std::collections::HashMap;
use crate::enumerations::enums::{
    category,quota,EngineeringBranch,IndianState,counselling,
};

pub type NodeId = usize;
pub type CandidateId = usize;

#[derive(Debug, Clone)]
pub struct SeatNode {
    pub id: NodeId,
    pub insti: String,
    pub branch: EngineeringBranch,
    pub quota: quota,
    pub category: category,
    pub max_seat_capacity: u16,
    pub pred_closing_rank: u32,
    pub counselling: counselling,
}


#[derive(Debug, Clone)]
pub struct CandidateNode {
    pub id: CandidateId,
    pub name: String,
    pub rank: u32,
    pub category: category,
    pub homestate: IndianState,
    pub preferences: Vec<(String, String)>, // Vec of (insti, branch)
    pub allocated_seat: Option<(String, String)>, // Option of (insti, branch)
}

#[derive(Debug, Clone)]
pub struct TransitionEdge{
    pub target: NodeId,
    pub probability: f32,
}


pub struct AdmissionGraph {
    pub seat_nodes: HashMap<NodeId, SeatNode>,
    pub candidate_nodes: HashMap<CandidateId, CandidateNode>,
    pub edges: HashMap<NodeId, Vec<TransitionEdge>>, // Adjacency list representation
}
