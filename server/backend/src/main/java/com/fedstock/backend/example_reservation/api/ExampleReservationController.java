package com.fedstock.backend.example_reservation.api;

import java.net.URI;
import java.util.List;

import jakarta.validation.Valid;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import com.fedstock.backend.example_reservation.api.dto.CreateExampleReservationRequest;
import com.fedstock.backend.example_reservation.api.dto.ExampleReservationResponse;
import com.fedstock.backend.example_reservation.application.ExampleReservationService;

@RestController
@RequestMapping("/api/example-reservations")
public class ExampleReservationController {

    private final ExampleReservationService reservationService;

    public ExampleReservationController(ExampleReservationService reservationService) {
        this.reservationService = reservationService;
    }

    @PostMapping
    public ResponseEntity<ExampleReservationResponse> create(
        @Valid @RequestBody CreateExampleReservationRequest request
    ) {
        ExampleReservationResponse response = ExampleReservationResponse.from(
            reservationService.create(request.toCommand())
        );
        URI location = ServletUriComponentsBuilder
            .fromCurrentRequest()
            .path("/{reservationId}")
            .buildAndExpand(response.id())
            .toUri();

        return ResponseEntity.created(location).body(response);
    }

    @GetMapping
    public List<ExampleReservationResponse> findAll() {
        return reservationService.findAll()
            .stream()
            .map(ExampleReservationResponse::from)
            .toList();
    }

    @GetMapping("/{reservationId}")
    public ExampleReservationResponse findById(@PathVariable Long reservationId) {
        return ExampleReservationResponse.from(reservationService.findById(reservationId));
    }

    @PatchMapping("/{reservationId}/confirm")
    public ExampleReservationResponse confirm(@PathVariable Long reservationId) {
        return ExampleReservationResponse.from(reservationService.confirm(reservationId));
    }

    @PatchMapping("/{reservationId}/cancel")
    public ExampleReservationResponse cancel(@PathVariable Long reservationId) {
        return ExampleReservationResponse.from(reservationService.cancel(reservationId));
    }
}
