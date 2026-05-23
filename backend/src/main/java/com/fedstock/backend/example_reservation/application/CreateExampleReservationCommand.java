package com.fedstock.backend.example_reservation.application;

import java.time.LocalDate;

public record CreateExampleReservationCommand(
    String customerName,
    LocalDate reservationDate,
    Integer guestCount
) {
}
