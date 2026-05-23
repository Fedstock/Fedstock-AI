package com.fedstock.backend.demo.application;

public record CreateDemoCommand(
    String title,
    String content
) {
}
