package com.fedstock.backend.example_reservation.domain;

import java.time.LocalDate;
import java.time.LocalDateTime;

public record ExampleReservation(
    Long id,
    String customerName,
    LocalDate reservationDate,
    int guestCount,
    ReservationStatus status,
    LocalDateTime createdAt,
    LocalDateTime updatedAt
) {
    public static ExampleReservation create(String customerName, LocalDate reservationDate, int guestCount) {
        LocalDateTime now = LocalDateTime.now();
        return new ExampleReservation(
            null,
            customerName,
            reservationDate,
            guestCount,
            ReservationStatus.REQUESTED,
            now,
            now
        );
    }

    public ExampleReservation assignId(Long id) {
        return new ExampleReservation(
            id,
            customerName,
            reservationDate,
            guestCount,
            status,
            createdAt,
            updatedAt
        );
    }

    public ExampleReservation confirm() {
        return changeStatus(ReservationStatus.CONFIRMED);
    }

    public ExampleReservation cancel() {
        return changeStatus(ReservationStatus.CANCELLED);
    }

    private ExampleReservation changeStatus(ReservationStatus nextStatus) {
        return new ExampleReservation(
            id,
            customerName,
            reservationDate,
            guestCount,
            nextStatus,
            createdAt,
            LocalDateTime.now()
        );
    }
}
