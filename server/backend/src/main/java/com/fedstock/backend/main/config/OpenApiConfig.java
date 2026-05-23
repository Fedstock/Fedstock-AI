package com.fedstock.backend.main.config;

import org.springframework.context.annotation.Configuration;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;

@Configuration
@OpenAPIDefinition(
    info = @Info(
        title = "Fedstock Backend API",
        version = "v1",
        description = "Fedstock backend API documentation"
    )
)
public class OpenApiConfig {
}
