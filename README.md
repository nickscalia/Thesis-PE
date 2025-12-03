# A Bio-inspired Sensor Fusion Approach Combining EMG, IMU, and Vision for Back-Exoskeleton Payload Estimation
*Master's Thesis - Mechanical Engineering (Mechatronics & Robotics)*  
Politecnico di Milano, 2025

---

## Author  
Nicolas Scalia - nicolas.scalia@mail.polimi.it

## Advisors  
Prof. Marta Gandolla   
Eng. Andrea Dal Prete (PhD Student) 

---

## Project Overview
This thesis focuses on improving payload estimation methods for back exoskeletons using sensor fusion. The project includes **online** and **offline** pipelines:  

- **Online:** integrates computer vision and EMG/IMU data to provide live payload estimates for industrial workers, with results communicated to a Raspberry Pi.  
- **Offline:** processes EMG-IMU data from Myo and Trigno sensors, as well as the full Myo and computer vision data acquired during online sessions, to analyse payloads and refine algorithms.

---

# Repository Structure

### `offline/`
- `myo_cv/`: Offline analysis of Myo and computer vision data acquired through the online system
  - `data/`: Acquisitions and processed data 
  - `src/`: Main offline analysis scripts
  - `environment.txt`: Dependencies for conda environment
 
- `myo_trigno/`: Offline analysis of EMG-IMU data from Myo and Trigno sensors  
  - `data/`: Acquisitions and processed data 
  - `lib/`: Utility functions and preprocessing scripts  
  - `src/`: Main offline analysis scripts
  - `environment.txt`: Dependencies for conda environment
 
### `online/`
- `com_vis/`: Computer vision-based payload estimation   
  - `models/`: Trained YOLO models  
  - `src/`: Main scripts for real-time CV estimation and sensor fusion
  - `requirements.txt`: Dependencies for virtual environment  

- `myo/`: EMG and IMU-based payload estimation  
  - `lib/`: Utility functions and preprocessing scripts  
  - `models/`: Trained ML models  
  - `src/`: Main scripts for real-time EMG/IMU estimation  
  - `requirements.txt`: Dependencies for virtual environment  

---

## Setup

Each pipeline has its own `requirements.txt`. It's recommended to create a **separate virtual environment** for each to install dependencies. 
