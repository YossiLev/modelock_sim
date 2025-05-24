# 🔦 Laser Simulation Web App

A Python-based web application for laser simulation, powered by [FastAPI](https://fastapi.tiangolo.com/) and served with [Uvicorn](https://www.uvicorn.org/).

## 🚀 Features

- Performs simulation of Gaussian beam in a cavity holding a **titanium sapphire crystal**. The simulation
follows the article: 
**Parshani, I; Bello, L; Meller, M; Pe'er, A**
[Kerr-Lens Mode-Locking: Numerical Simulation of the Spatio-Temporal Dynamics on All Time Scales](https://doi.org/10.3390/app122010354).

- Performs simulation of a multimode beam in a cavity holding a titanium sapphire crystal.

- Performs various calculations of Gaussian Laser beam stability in a cavity.

## 📦 Installation

Code tested with python 3.11.6 so we recommend python 3.11

Found incompatible with 3.12 3.13 so please avoid using those version for this project

```bash
# clone the repo
git clone https://github.com/YossiLev/modelock_sim

# navigate into the project folder
cd modelock_sim

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Activation
Run the developement server

- From the project folder run:

uvicorn app:app --host 0.0.0.0 --port 8000

- From your browser go to

http://localhost:8000


## 📁 Project Structure
```
laser-sim/
├── app.py                             # Uvicorn entry point
├── other_application_modules.py       # Application modules
├── static/                            # Frontend assets
├── requirements.txt
└── README.md
```

## 📄 License
Copyright (c) 2024-2025, Bar Ilan University, Prof. Avi Pe'er Lab

This project is licensed under the [MIT License](https://opensource.org/license/mit).

## 🙋‍♂️ Contact
For questions or feedback, feel free to contact me at **yossi.lev.home@gmail.com**.