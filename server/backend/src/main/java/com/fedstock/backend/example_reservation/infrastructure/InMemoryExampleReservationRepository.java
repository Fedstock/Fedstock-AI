package com.fedstock.backend.example_reservation.infrastructure;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import org.springframework.stereotype.Repository;

import com.fedstock.backend.example_reservation.domain.ExampleReservation;
import com.fedstock.backend.example_reservation.domain.ExampleReservationRepository;

@Repository
public class InMemoryExampleReservationRepository implements ExampleReservationRepository {

    private final AtomicLong sequence = new AtomicLong();
    private final Map<Long, ExampleReservation> reservations = new ConcurrentHashMap<>();

    @Override
    public ExampleReservation save(ExampleReservation reservation) {
        ExampleReservation saved = reservation.id() == null
            ? reservation.assignId(sequence.incrementAndGet())
            : reservation;

        reservations.put(saved.id(), saved);
        return saved;
    }

    @Override
    public Optional<ExampleReservation> findById(Long reservationId) {
        return Optional.ofNullable(reservations.get(reservationId));
    }

    @Override
    public List<ExampleReservation> findAll() {
        return reservations.values()
            .stream()
            .sorted(Comparator.comparing(ExampleReservation::id))
            .toList();
    }
}
