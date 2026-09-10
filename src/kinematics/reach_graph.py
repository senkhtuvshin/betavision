"""Reachability graph over detected holds and A* search for an optimal beta path.

Coordinates throughout are assumed to be in the same unit as the hold centroids
(typically image pixels). `ClimberProfile.max_reach_radius` must be expressed in
that same unit for reach constraints to be meaningful.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.kinematics.pose_tracker import ClimberPose
from src.vision.hold_detector import HoldDetection

START_NODE_ID = -1

# Limb extremities checked for hold contact.
CONTACT_LIMBS: tuple[str, ...] = ("LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE")
DEFAULT_CONTACT_RADIUS = 50.0  # px fallback if the caller doesn't scale one to the climber's reach

# Weight factors for the edge-cost heuristic; tuned by feel rather than measurement.
LATERAL_COM_PENALTY = 0.6  # cost per unit of sideways CoM shift between holds
DOWNWARD_MOVE_PENALTY = 1.5  # cost per unit of downward movement (discourages backtracking)
NEAR_MAX_REACH_PENALTY = 40.0  # extra cost scaling with (reach / max_reach)^2


@dataclass
class ClimberProfile:
    """Anthropometric constraints used to decide which hold-to-hold reaches are valid."""

    height: float
    arm_span: float
    max_reach_radius: float | None = None

    def __post_init__(self) -> None:
        if self.max_reach_radius is None:
            # A climber can typically reach a bit beyond half their wingspan via hip/shoulder rotation.
            self.max_reach_radius = self.arm_span * 0.55


@dataclass
class HoldNode:
    """A graph node wrapping a detected hold (or the climber's starting position)."""

    id: int
    centroid: tuple[float, float]
    area: float = 0.0
    detection: HoldDetection | None = None


@dataclass
class ReachabilityGraph:
    """Graph of holds connected by edges the climber can physically reach between."""

    climber: ClimberProfile
    nodes: dict[int, HoldNode] = field(default_factory=dict)
    edges: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        holds: list[HoldDetection],
        climber: ClimberProfile,
        start_position: tuple[float, float] | None = None,
    ) -> "ReachabilityGraph":
        """Construct the graph from detected holds, optionally seeded with the climber's
        current position as a virtual start node (id=START_NODE_ID) with only outgoing edges.
        """
        nodes = {
            i: HoldNode(id=i, centroid=hold.centroid, area=hold.area, detection=hold)
            for i, hold in enumerate(holds)
        }
        if start_position is not None:
            nodes[START_NODE_ID] = HoldNode(id=START_NODE_ID, centroid=start_position)

        graph = cls(climber=climber, nodes=nodes)
        graph.edges = graph._build_edges(virtual_start_id=START_NODE_ID if start_position is not None else None)
        return graph

    def _build_edges(self, virtual_start_id: int | None) -> dict[int, list[tuple[int, float]]]:
        edges: dict[int, list[tuple[int, float]]] = {node_id: [] for node_id in self.nodes}

        for a_id, a in self.nodes.items():
            if a_id == virtual_start_id:
                continue  # the virtual start node has no incoming edges, handled below
            for b_id, b in self.nodes.items():
                if a_id == b_id or b_id == virtual_start_id:
                    continue
                cost = edge_cost(a, b, self.climber)
                if cost is not None:
                    edges[a_id].append((b_id, cost))

        if virtual_start_id is not None:
            start = self.nodes[virtual_start_id]
            for node_id, node in self.nodes.items():
                if node_id == virtual_start_id:
                    continue
                cost = edge_cost(start, node, self.climber)
                if cost is not None:
                    edges[virtual_start_id].append((node_id, cost))

        return edges

    def neighbors(self, node_id: int) -> list[tuple[int, float]]:
        return self.edges.get(node_id, [])


def edge_cost(a: HoldNode, b: HoldNode, climber: ClimberProfile) -> float | None:
    """Cost of moving from hold `a` to hold `b`, or None if it exceeds the climber's reach.

    Penalizes reach distance itself, sideways CoM displacement, downward movement, and
    reaches that approach the climber's maximum reach radius (which are less controlled).
    """
    dx = b.centroid[0] - a.centroid[0]
    dy = b.centroid[1] - a.centroid[1]
    distance = math.hypot(dx, dy)

    if distance > climber.max_reach_radius:
        return None

    lateral_penalty = abs(dx) * LATERAL_COM_PENALTY
    downward_penalty = max(0.0, dy) * DOWNWARD_MOVE_PENALTY  # image y grows downward
    reach_ratio = distance / climber.max_reach_radius
    strain_penalty = (reach_ratio**2) * NEAR_MAX_REACH_PENALTY

    return distance + lateral_penalty + downward_penalty + strain_penalty


def find_contact_holds(
    pose: ClimberPose,
    holds: list[HoldDetection],
    contact_radius: float = DEFAULT_CONTACT_RADIUS,
) -> dict[str, int | None]:
    """Map each limb extremity (wrists, ankles) to the index of the hold it's grounded on.

    A limb is considered "in contact" with whichever hold is nearest its pixel position, as
    long as that distance is within `contact_radius`; otherwise it maps to None (e.g. a hand
    mid-reach, not yet gripping anything).
    """
    contacts: dict[str, int | None] = {}

    for limb in CONTACT_LIMBS:
        limb_position = pose.pixel[limb]
        nearest_idx: int | None = None
        nearest_distance = math.inf

        for i, hold in enumerate(holds):
            distance = math.dist(limb_position, hold.centroid)
            if distance < nearest_distance:
                nearest_idx, nearest_distance = i, distance

        contacts[limb] = nearest_idx if nearest_distance <= contact_radius else None

    return contacts


def find_beta_path(graph: ReachabilityGraph, start_id: int | list[int], goal_id: int) -> list[int] | None:
    """A* search for the lowest-cost sequence of holds from start_id (or several, for a
    multi-limb-grounded start) to goal_id.
    """
    start_ids = start_id if isinstance(start_id, list) else [start_id]
    if not start_ids or any(sid not in graph.nodes for sid in start_ids) or goal_id not in graph.nodes:
        raise KeyError("start_id(s) and goal_id must all be nodes in the graph")

    def heuristic(node_id: int) -> float:
        return math.dist(graph.nodes[node_id].centroid, graph.nodes[goal_id].centroid)

    tie_breaker = itertools.count()
    open_heap = [(heuristic(sid), next(tie_breaker), sid) for sid in start_ids]
    heapq.heapify(open_heap)
    came_from: dict[int, int] = {}
    g_score = {sid: 0.0 for sid in start_ids}
    visited: set[int] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == goal_id:
            return _reconstruct_path(came_from, current)
        visited.add(current)

        for neighbor_id, cost in graph.neighbors(current):
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor_id, math.inf):
                g_score[neighbor_id] = tentative_g
                came_from[neighbor_id] = current
                f_score = tentative_g + heuristic(neighbor_id)
                heapq.heappush(open_heap, (f_score, next(tie_breaker), neighbor_id))

    return None


def _reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def draw_beta_path(
    frame: np.ndarray,
    graph: ReachabilityGraph,
    path: list[int],
    line_color: tuple[int, int, int] = (60, 220, 130),
    start_color: tuple[int, int, int] = (0, 165, 255),
    goal_color: tuple[int, int, int] = (0, 0, 255),
    glow_radius: int = 6,
) -> np.ndarray:
    """Return a copy of `frame` with a glowing trajectory line and numbered move markers.

    The trajectory is drawn twice: a thick, Gaussian-blurred "glow" pass underneath, then a
    crisp core line on top. Start and goal holds get distinctive marker colors; intermediate
    moves are numbered in sequence.
    """
    overlay = frame.copy()
    points = [tuple(round(v) for v in graph.nodes[node_id].centroid) for node_id in path]

    if len(points) > 1:
        glow_layer = np.zeros_like(frame)
        for p1, p2 in zip(points, points[1:]):
            cv2.line(glow_layer, p1, p2, line_color, glow_radius * 2, cv2.LINE_AA)
        glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=glow_radius)
        overlay = cv2.addWeighted(overlay, 1.0, glow_layer, 0.7, 0)

        for p1, p2 in zip(points, points[1:]):
            cv2.line(overlay, p1, p2, line_color, 2, cv2.LINE_AA)

    for step, (node_id, point) in enumerate(zip(path, points)):
        if node_id == path[0]:
            marker_color = start_color
        elif node_id == path[-1]:
            marker_color = goal_color
        else:
            marker_color = line_color

        cv2.circle(overlay, point, 13, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, point, 11, marker_color, -1, cv2.LINE_AA)

        label = str(step + 1)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(
            overlay,
            label,
            (point[0] - text_w // 2, point[1] + text_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay
