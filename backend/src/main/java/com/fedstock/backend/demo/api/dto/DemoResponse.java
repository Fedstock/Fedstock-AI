package com.fedstock.backend.demo.api.dto;

import java.time.LocalDateTime;

import com.fedstock.backend.demo.domain.Demo;

public record DemoResponse(
    Long id,
    String title,
    String content,
    LocalDateTime createdAt,
    LocalDateTime updatedAt
) {
    public static DemoResponse from(Demo demo) {
        return new DemoResponse(
            demo.id(),
            demo.title(),
            demo.content(),
            demo.createdAt(),
            demo.updatedAt()
        );
    }
}
