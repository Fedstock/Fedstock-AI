package com.fedstock.backend.demo.domain;

import java.util.List;
import java.util.Optional;

public interface DemoRepository {

    Demo save(Demo demo);

    Optional<Demo> findById(Long demoId);

    List<Demo> findAll();

    void deleteById(Long demoId);
}
