package com.fedstock.backend.example_reservation.api.dto;

import java.time.LocalDate;
import java.time.LocalDateTime;

import com.fedstock.backend.example_reservation.domain.ExampleReservation;
import com.fedstock.backend.example_reservation.domain.ReservationStatus;

public record ExampleReservationResponse(
    Long id,
    String customerName,
    LocalDate reservationDate,
    int guestCount,
    ReservationStatus status,
    LocalDateTime createdAt,
    LocalDateTime updatedAt
) {
    public static ExampleReservationResponse from(ExampleReservation reservation) {
        return new ExampleReservationResponse(
            reservation.id(),
            reservation.customerName(),
            reservation.reservationDate(),
            reservation.guestCount(),
            reservation.status(),
            reservation.createdAt(),
            reservation.updatedAt()
        );
    }
}
