package com.fedstock.backend.demo.infrastructure;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Repository;

import com.fedstock.backend.demo.domain.Demo;
import com.fedstock.backend.demo.domain.DemoRepository;

@Repository
public class DemoRepositoryAdapter implements DemoRepository {

    private final DemoJpaRepository demoJpaRepository;

    public DemoRepositoryAdapter(DemoJpaRepository demoJpaRepository) {
        this.demoJpaRepository = demoJpaRepository;
    }

    @Override
    public Demo save(Demo demo) {
        return demoJpaRepository.save(DemoEntity.from(demo)).toDomain();
    }

    @Override
    public Optional<Demo> findById(Long demoId) {
        return demoJpaRepository.findById(demoId).map(DemoEntity::toDomain);
    }

    @Override
    public List<Demo> findAll() {
        return demoJpaRepository.findAll()
            .stream()
            .map(DemoEntity::toDomain)
            .toList();
    }

    @Override
    public void deleteById(Long demoId) {
        demoJpaRepository.deleteById(demoId);
    }
}
