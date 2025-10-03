Dataset Organization
This repository contains data organized from various acquisition sessions using different devices. Below is an overview of the folder structure and naming conventions used:

Folder Structure
_media/: Contains videos recorded during the acquisition sessions.
myo/: Contains acquisition data collected using the Myo armband.
trigno/: Contains acquisition data collected using the Delsys Trigno system.

Each of these folders is further organized by date, using the format: month_day (e.g., 05_14 for May 14th).

Within each date folder, data is categorized based on the weight/intensity level of the task performed:
_mvc/ - MVC measurements
heavy/ – Heavy tasks
medium/ – Medium tasks
light/ – Light tasks
all/ – All tasks combined

File Naming Convention
Each file follows a standardized naming convention:
S<subject_number>_<intensity><acquisition_number>_<additional_letter>

Where:
SXX = Subject identifier (e.g., S01, S02, ...)
A, H, M, L, X = Task intensity: All, Heavy, Medium, Light, MVC measurement
01, 02, ... = Acquisition number for that specific subject and intensity level
O, BC, TC, FA1, FA2 = Additional Letter: Motion, Biceps, Triceps, Forearm 1, Forearm 2

Subjects:
01 -> Nicolas