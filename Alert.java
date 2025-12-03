package com.ids.model;


import jakarta.persistence.*;
import java.time.LocalDateTime;


@Entity
public class Alert {
@Id
@GeneratedValue(strategy = GenerationType.IDENTITY)
private Long id;
private String message;
private String level; // INFO, WARNING, CRITICAL
private LocalDateTime timestamp;


public Alert() {}
public Alert(String message, String level) {
this.message = message;
this.level = level;
this.timestamp = LocalDateTime.now();
}


// getters/setters
public Long getId() { return id; }
public void setId(Long id) { this.id = id; }
public String getMessage() { return message; }
public void setMessage(String message) { this.message = message; }
public String getLevel() { return level; }
public void setLevel(String level) { this.level = level; }
public LocalDateTime getTimestamp() { return timestamp; }
public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
}