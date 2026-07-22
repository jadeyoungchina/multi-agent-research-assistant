def route_after_critic(
    iteration: int, sufficient: bool, max_iterations: int
) -> tuple[str, int]:
    """Choose the next workflow stage without changing the critic's verdict."""
    if sufficient:
        return "writer", iteration
    if iteration < max_iterations:
        return "retriever", iteration + 1
    return "writer", iteration
