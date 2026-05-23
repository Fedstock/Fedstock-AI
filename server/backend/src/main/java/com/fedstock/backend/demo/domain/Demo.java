package com.fedstock.backend.demo.domain;

import java.time.LocalDateTime;

public record Demo(
    Long id,
    String title,
    String content,
    LocalDateTime createdAt,
    LocalDateTime updatedAt
) {
    public static Demo create(String title, String content) {
        return new Demo(null, title, content, null, null);
    }

    public Demo update(String title, String content) {
        return new Demo(id, title, content, createdAt, updatedAt);
    }
}
