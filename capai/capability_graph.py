"""
capai.capability_graph
=========================
Capability Graphs (research extension #3): tracks declared dependencies
between capabilities, so that CapAI can, in principle, compose a new
capability from existing verified building blocks instead of asking an
LLM to write everything from scratch.

IMPORTANT — what this is and is not:
This module tracks and queries a DECLARED dependency graph (edges are
recorded when a capability's synthesis prompt or specification names
another existing capability as a building block, or when a human/agent
explicitly registers a dependency). It does NOT perform automatic
program-level dependency inference from source code (e.g. static call-
graph analysis), and it does NOT yet attempt automatic composition of a
new capability's implementation from its declared dependencies' source
— that composition step (turning a resolved dependency chain into a
working, verified function that calls its dependencies rather than
reimplementing their logic) is flagged as future work in the roadmap
below, not claimed as solved here.

What IS real and working: a directed acyclic graph structure, cycle
detection, dependency resolution (topological ordering), and a "what
would break if I removed this" impact-analysis query — all of which
are useful today even before automatic composition is built, e.g. for
safe deprecation of a capability that others depend on.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


class CycleError(Exception):
    """Raised when adding a dependency edge would create a cycle."""


@dataclass
class CapabilityGraph:
    # name -> set of names it depends on
    edges: dict = field(default_factory=lambda: defaultdict(set))
    # name -> set of names that depend on it (reverse index, kept in sync)
    reverse_edges: dict = field(default_factory=lambda: defaultdict(set))

    def add_dependency(self, capability: str, depends_on: str) -> None:
        """
        Declare that `capability` depends on `depends_on`. Raises
        CycleError if this would create a cycle (e.g. A depends on B,
        B depends on A) rather than silently corrupting the graph.
        """
        if capability == depends_on:
            raise CycleError(f"'{capability}' cannot depend on itself")

        # temporarily add and check for a cycle via DFS before committing
        self.edges[capability].add(depends_on)
        if self._has_cycle_from(capability):
            self.edges[capability].discard(depends_on)
            raise CycleError(
                f"Adding '{capability}' -> '{depends_on}' would create a cycle"
            )
        self.reverse_edges[depends_on].add(capability)

    def _has_cycle_from(self, start: str) -> bool:
        visited, stack = set(), [start]
        path = set()

        def dfs(node):
            if node in path:
                return True
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for neighbour in self.edges.get(node, ()):
                if dfs(neighbour):
                    return True
            path.discard(node)
            return False

        return dfs(start)

    def dependencies_of(self, capability: str) -> set:
        """Direct dependencies only (one hop)."""
        return set(self.edges.get(capability, ()))

    def all_dependencies_of(self, capability: str) -> set:
        """Transitive closure — every capability this one depends on, directly or indirectly."""
        seen, queue = set(), deque([capability])
        while queue:
            current = queue.popleft()
            for dep in self.edges.get(current, ()):
                if dep not in seen:
                    seen.add(dep)
                    queue.append(dep)
        return seen

    def dependents_of(self, capability: str) -> set:
        """Everything that directly depends on this capability."""
        return set(self.reverse_edges.get(capability, ()))

    def impact_of_removing(self, capability: str) -> set:
        """
        Transitive closure of everything that would be affected if
        `capability` were deleted or changed — i.e. the full blast
        radius, not just direct dependents. Useful for safe deprecation:
        "what breaks if I delete calculate_percentage?"
        """
        seen, queue = set(), deque([capability])
        while queue:
            current = queue.popleft()
            for dependent in self.reverse_edges.get(current, ()):
                if dependent not in seen:
                    seen.add(dependent)
                    queue.append(dependent)
        return seen

    def topological_build_order(self, capability: str) -> list:
        """
        Returns the order in which `capability`'s dependencies would
        need to be built first, ending with `capability` itself — the
        order an automatic-composition step (future work) would need
        to synthesize or verify things in.
        """
        order = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in self.edges.get(node, ()):
                visit(dep)
            order.append(node)

        visit(capability)
        return order

    def to_dict(self) -> dict:
        """Serializable form for persisting the graph to MongoDB."""
        return {k: sorted(v) for k, v in self.edges.items() if v}
