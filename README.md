# Payload Estimation for Back-Exoskeleton Control  
*Master's Thesis - Mechanical Engineering*  
Politecnico di Milano, 2025

---

## Author  
Nicolas Scalia - nicolas.scalia@mail.polimi.it

## Advisors  
Prof. Marta Gandolla   
Andrea Dal Prete (PhD Student) 

---

## Project Overview
This thesis focuses on improving payload estimation methods for back exoskeletons using sensor fusion. The project includes **real-time** and **offline** pipelines:  

- **Real-time:** integrates computer vision and EMG/IMU data to provide live payload estimates for industrial workers, with results communicated to a Raspberry Pi.  
- **Offline:** processes EMG/IMU data from Myo and Trigno sensors to analyze payloads and refine algorithms.  

---

# Repository Structure

### `real_time/`
- `comp_vis/`: Computer vision-based payload estimation  
  - `data/`: Session-specific datasets  
  - `models/`: Trained vision models  
  - `src/`: Main code for real-time CV estimation  
  - `requirements.txt`: Dependencies for virtual environment  

- `myo/`: EMG and IMU-based payload estimation  
  - `data/`: Session-specific datasets  
  - `lib/`: Utility functions and preprocessing scripts  
  - `models/`: Trained ML models  
  - `src/`: Main code for real-time EMG/IMU estimation  
  - `requirements.txt`: Dependencies for virtual environment  

- `shared/`: Temporary files used to combine vision and EMG/IMU estimates for the final payload sent to the Raspberry Pi  

### `offline/`
- `myo_trigno/`: Offline analysis of EMG/IMU data from Myo and Trigno sensors  
  - `data/`: Acquisitions and processed datasets  
  - `lib/`: Utility functions and preprocessing scripts  
  - `src/`: Main offline analysis code
  - `environment.yml`: Dependencies for conda environment  

---

## Setup

Each real_time pipeline has its own `requirements.txt`. It's recommended to create a **separate virtual environment** for each to install dependencies. 
The offline pipeline uses Jupyter Notebook via Anaconda. Create a conda environment and install the dependencies using the provided `environment.yml`.  
