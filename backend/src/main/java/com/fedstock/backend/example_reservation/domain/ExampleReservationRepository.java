package com.fedstock.backend.example_reservation.domain;

import java.util.List;
import java.util.Optional;

public interface ExampleReservationRepository {

    ExampleReservation save(ExampleReservation reservation);

    Optional<ExampleReservation> findById(Long reservationId);

    List<ExampleReservation> findAll();
}
