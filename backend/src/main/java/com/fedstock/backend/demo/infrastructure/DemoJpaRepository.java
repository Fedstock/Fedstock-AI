package com.fedstock.backend.demo.infrastructure;

import org.springframework.data.jpa.repository.JpaRepository;

interface DemoJpaRepository extends JpaRepository<DemoEntity, Long> {
}
