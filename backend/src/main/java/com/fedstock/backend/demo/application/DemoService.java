package com.fedstock.backend.demo.application;

import java.util.List;
import java.util.NoSuchElementException;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.fedstock.backend.demo.domain.Demo;
import com.fedstock.backend.demo.domain.DemoRepository;

@Service
@Transactional
public class DemoService {

    private final DemoRepository demoRepository;

    public DemoService(DemoRepository demoRepository) {
        this.demoRepository = demoRepository;
    }

    public Demo create(CreateDemoCommand command) {
        return demoRepository.save(Demo.create(command.title(), command.content()));
    }

    @Transactional(readOnly = true)
    public List<Demo> findAll() {
        return demoRepository.findAll();
    }

    @Transactional(readOnly = true)
    public Demo findById(Long demoId) {
        return getDemo(demoId);
    }

    public Demo update(Long demoId, UpdateDemoCommand command) {
        Demo demo = getDemo(demoId).update(command.title(), command.content());
        return demoRepository.save(demo);
    }

    public void delete(Long demoId) {
        getDemo(demoId);
        demoRepository.deleteById(demoId);
    }

    private Demo getDemo(Long demoId) {
        return demoRepository.findById(demoId)
            .orElseThrow(() -> new NoSuchElementException("Demo not found: " + demoId));
    }
}
