# -----------------------------------------------------------------------------
# Copyright (c) 2025 Nicolas Scalia - Politecnico di Milano
# All rights reserved.
#
# This script is part of the research published in:
# [Your Paper Title], [Conference/Journal Name], [Year]
# DOI: [Insert DOI if available]
#
# Author: Nicolas Scalia (nicolas.scalia@mail.polimi.it)
# -----------------------------------------------------------------------------

#%% CODE EXPLAINATION 
# This script launches the EMG Payload Estimator GUI by creating a window and instantiating EMGApp. 
# Starts the main loop to interact with the user. 
# Keeps core logic separated in emg_app.py.

# Necessary libraries
import ttkbootstrap as ttk
from emg_app import EMGApp

#%% Creates the GUI window and starts the EMG Payload Estimator application.
if __name__ == "__main__":
    root = ttk.Window(themename="flatly") 
    app = EMGApp(root)
    root.mainloop()