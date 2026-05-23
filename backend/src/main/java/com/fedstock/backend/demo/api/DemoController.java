package com.fedstock.backend.demo.api;

import java.net.URI;
import java.util.List;

import jakarta.validation.Valid;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import com.fedstock.backend.demo.api.dto.CreateDemoRequest;
import com.fedstock.backend.demo.api.dto.DemoResponse;
import com.fedstock.backend.demo.api.dto.UpdateDemoRequest;
import com.fedstock.backend.demo.application.DemoService;

@RestController
@RequestMapping("/api/demos")
public class DemoController {

    private final DemoService demoService;

    public DemoController(DemoService demoService) {
        this.demoService = demoService;
    }

    @PostMapping
    public ResponseEntity<DemoResponse> create(@Valid @RequestBody CreateDemoRequest request) {
        DemoResponse response = DemoResponse.from(demoService.create(request.toCommand()));
        URI location = ServletUriComponentsBuilder
            .fromCurrentRequest()
            .path("/{demoId}")
            .buildAndExpand(response.id())
            .toUri();

        return ResponseEntity.created(location).body(response);
    }

    @GetMapping
    public List<DemoResponse> findAll() {
        return demoService.findAll()
            .stream()
            .map(DemoResponse::from)
            .toList();
    }

    @GetMapping("/{demoId}")
    public DemoResponse findById(@PathVariable Long demoId) {
        return DemoResponse.from(demoService.findById(demoId));
    }

    @PutMapping("/{demoId}")
    public DemoResponse update(
        @PathVariable Long demoId,
        @Valid @RequestBody UpdateDemoRequest request
    ) {
        return DemoResponse.from(demoService.update(demoId, request.toCommand()));
    }

    @DeleteMapping("/{demoId}")
    public ResponseEntity<Void> delete(@PathVariable Long demoId) {
        demoService.delete(demoId);
        return ResponseEntity.noContent().build();
    }
}
