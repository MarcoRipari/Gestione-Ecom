def estimate_tokens(prompt: str) -> tuple[int, float]:
    num_chars = len(prompt)
    token_est = num_chars // 4
    cost_est = round(token_est / 1000 * 0.001, 6)
    return token_est, cost_est
