import pytest

from app.workflow.routing import route_after_critic


@pytest.mark.parametrize(
    ("iteration", "sufficient", "expected_stage", "expected_iteration"),
    [
        (0, False, "retriever", 1),
        (1, False, "retriever", 2),
        (2, False, "writer", 2),
        (0, True, "writer", 0),
    ],
)
def test_route_after_critic(
    iteration: int,
    sufficient: bool,
    expected_stage: str,
    expected_iteration: int,
) -> None:
    assert route_after_critic(iteration, sufficient, max_iterations=2) == (
        expected_stage,
        expected_iteration,
    )
