package com.fedstock.backend.main.api;

import java.time.LocalDateTime;

public record MainHealthResponse(
    String status,
    LocalDateTime checkedAt
) {
}
