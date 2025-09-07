# Laser Quality Assessor
This project was done as a part of the requirements for the course Intelligent Data Analysis & Machine Learning I, offered at the University of Potsdam, by the Institut für Informatik.

## Instructions

### Requirements:
1. Python 3.11.1 or later
1. Git
1. Jupyter Labs or equivalent

To run the project:
1. Clone the repo using SSH or HTTPS from GitUP or the GitHub mirror: <br>
    **GitUP:** <br>
    SSH
    ```
    git clone git@gitup.uni-potsdam.de:ansari/laserqualityassessor.git
    ```
    HTTPS
    ```
    git clone https://gitup.uni-potsdam.de/ansari/laserqualityassessor.git
    ```
    **GitHub:**<br>
    SSH
    ```
    git clone git@github.com:cosmonautinspace/laserQualityAsessor.git
    ```
    HTTPS
    ```
    git clone https://github.com/cosmonautinspace/laserQualityAsessor.git
    ```

1. (Optional) It is highly recommend to use a virtual environment. Basic instructions are provided here for the standard python `venv` package but `conda` can also be used. A more extensive documentation of the venv package can be found [here](https://docs.python.org/3/library/venv.html).
    1. Create a virtual environment in the folder you wish to run the project in by running:
        - On Windows (powershell/terminal):
        ```
        python -m venv <env_name>
        ```
        - On linux (bash):
        ```
        python3 -m venv <env_name>
        ```
        **NOTE:** it is recommended to create create the virtual environement in the project folder. This makes it easier for certain notebook environments to auto-detect installed python kernals.
    1. Activate the environment by running:
        - On Windows (powershell/terminal):
        ```
        <env_name>/Scripts/activate
        ```
        - On linux (bash):
        ```
        source <env_name>/bin/activate
        ```
        - For other shell variants, please refer to the `venv` documentation linked above.
    1. (Optional) Update pip by running:
        ```
        python -m pip install --upgrade pip
        ```
    1. From the environment, change directory to that of the cloned repo
    1. Install the required packages by running. This step can take a while, depending on the network connection, please be patient.:
        ```
        pip install -r requirements.txt
        ```

1. Now, open the cloned repository, in a Notebook environment of your choice, and make sure to select the correct python enviroment, if one was made using the instructions above. If you are not using virtual environments, please make sure to install the packages specified in the requirements.txt . It might also be required to install the package `ipykernal` inside the environment. Some Notebook environments, allow you to install this with the click of a button, while others might not. You might have to look up instructions for this elsewhere, as they are not provided in this document.

1. Run the following Jupyter notebooks, found under `src`
    - `eda.ipynb`
    - `baseline.ipynb`
    - `kernel_methods.ipynb`


## Ownership:
Name: Haani Ansari <br>
Email: haani.ansari@uni-potsdam.de