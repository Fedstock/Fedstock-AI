package com.fedstock.backend.demo.api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import com.fedstock.backend.demo.application.UpdateDemoCommand;

public record UpdateDemoRequest(
    @NotBlank(message = "title is required.")
    @Size(max = 100, message = "title must be 100 characters or less.")
    String title,

    @NotBlank(message = "content is required.")
    @Size(max = 1000, message = "content must be 1000 characters or less.")
    String content
) {
    public UpdateDemoCommand toCommand() {
        return new UpdateDemoCommand(title, content);
    }
}
