# ParlorTTS Web App

A local text-to-speech app with automated emotion detection.

---

## 1. Prerequisites

You must have **Python 3.10 ~ 3.13** or **Python Install Manager** installed on your computer

1. Download it from [python.org/downloads](https://www.python.org/downloads/)
* **Note:** Despite Python Install Manager installing with the version 3.14, this app can still run regardless
> This is because Python Install Manager comes with installation management, allowing other versions to be installed on command, which this setup will use automatically for its execution

---

## 2. How to Run

### Quick Start
Just double-click **`run.bat`**.

* **First-time launch:** It detects that setup hasn't been run yet, automatically installs all required packages, generates the voice previews, and pops open your web browser to `http://127.0.0.1:7860`. *(This first run takes a few minutes to download the AI models)*
* **Future launches:** Double-clicking `run.bat` starts the web app immediately

---

## 3. Troubleshooting & Notes

* **"Python was not found" error:** Python is either not installed or was not added to your system PATH. Re-run the Python installer, select **Modify**, and check **Add Python to environment variables**
* **Generation takes a bit of time:** The speech model runs locally on your machine's processor. Generating an audio clip may take 10–25 seconds depending on hardware; keep sentences short for faster results
* **Force a fresh reinstall:** If packages fail or you want to restart from scratch, delete the `.setup_done` file in the project folder and double-click `run.bat` again