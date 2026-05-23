package com.fedstock.backend.demo.application;

public record UpdateDemoCommand(
    String title,
    String content
) {
}
