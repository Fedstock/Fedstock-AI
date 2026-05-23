package com.fedstock.backend.example_reservation.application;

import java.util.List;
import java.util.NoSuchElementException;
import java.util.function.UnaryOperator;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.fedstock.backend.example_reservation.domain.ExampleReservation;
import com.fedstock.backend.example_reservation.domain.ExampleReservationRepository;

@Service
@Transactional
public class ExampleReservationService {

    private final ExampleReservationRepository reservationRepository;

    public ExampleReservationService(ExampleReservationRepository reservationRepository) {
        this.reservationRepository = reservationRepository;
    }

    public ExampleReservation create(CreateExampleReservationCommand command) {
        ExampleReservation reservation = ExampleReservation.create(
            command.customerName(),
            command.reservationDate(),
            command.guestCount()
        );

        return reservationRepository.save(reservation);
    }

    @Transactional(readOnly = true)
    public List<ExampleReservation> findAll() {
        return reservationRepository.findAll();
    }

    @Transactional(readOnly = true)
    public ExampleReservation findById(Long reservationId) {
        return getReservation(reservationId);
    }

    public ExampleReservation confirm(Long reservationId) {
        return update(reservationId, ExampleReservation::confirm);
    }

    public ExampleReservation cancel(Long reservationId) {
        return update(reservationId, ExampleReservation::cancel);
    }

    private ExampleReservation update(Long reservationId, UnaryOperator<ExampleReservation> updater) {
        ExampleReservation reservation = getReservation(reservationId);
        return reservationRepository.save(updater.apply(reservation));
    }

    private ExampleReservation getReservation(Long reservationId) {
        return reservationRepository.findById(reservationId)
            .orElseThrow(() -> new NoSuchElementException("Reservation not found: " + reservationId));
    }
}
