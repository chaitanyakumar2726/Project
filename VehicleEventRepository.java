package com.ids.repository;


import com.ids.model.VehicleEvent;
import org.springframework.data.jpa.repository.JpaRepository;


public interface VehicleEventRepository extends JpaRepository<VehicleEvent, Long> { }