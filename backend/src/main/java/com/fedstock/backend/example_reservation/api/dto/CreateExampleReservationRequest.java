package com.fedstock.backend.example_reservation.api.dto;

import java.time.LocalDate;

import jakarta.validation.constraints.FutureOrPresent;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

import com.fedstock.backend.example_reservation.application.CreateExampleReservationCommand;

public record CreateExampleReservationRequest(
    @NotBlank(message = "customerName is required.")
    @Size(max = 30, message = "customerName must be 30 characters or less.")
    String customerName,

    @NotNull(message = "reservationDate is required.")
    @FutureOrPresent(message = "reservationDate must be today or later.")
    LocalDate reservationDate,

    @NotNull(message = "guestCount is required.")
    @Min(value = 1, message = "guestCount must be at least 1.")
    @Max(value = 20, message = "guestCount must be 20 or less.")
    Integer guestCount
) {
    public CreateExampleReservationCommand toCommand() {
        return new CreateExampleReservationCommand(customerName, reservationDate, guestCount);
    }
}
