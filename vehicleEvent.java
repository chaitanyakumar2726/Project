package com.ids.model;


import jakarta.persistence.*;
import java.time.LocalDateTime;


@Entity
public class VehicleEvent {
@Id
@GeneratedValue(strategy = GenerationType.IDENTITY)
private Long id;


private String eventType; // INTRUSION, ANOMALY, AUTH_FAIL
private String description;
private double latitude;
private double longitude;
private LocalDateTime timestamp;


// constructors, getters, setters
public VehicleEvent() { }


public VehicleEvent(String eventType, String description, double latitude, double longitude) {
this.eventType = eventType;
this.description = description;
this.latitude = latitude;
this.longitude = longitude;
this.timestamp = LocalDateTime.now();
}


// getters/setters omitted for brevity - include them in actual file
public Long getId() { return id; }
public void setId(Long id) { this.id = id; }
public String getEventType() { return eventType; }
public void setEventType(String eventType) { this.eventType = eventType; }
public String getDescription() { return description; }
public void setDescription(String description) { this.description = description; }
public double getLatitude() { return latitude; }
public void setLatitude(double latitude) { this.latitude = latitude; }
public double getLongitude() { return longitude; }
public void setLongitude(double longitude) { this.longitude = longitude; }
public LocalDateTime getTimestamp() { return timestamp; }
public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
}