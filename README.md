"# Spatiotemporal-Routing-Algorithm-with-Changing-Conditions" 

## Instalation guide
### 1. To run this program you have to install the required packages with 
` pip install -r requirements.txt `

### 2. Create a .env file which will contain the path to the folder in which you downloaded the folder

It should look something like this

` BASE_DATA_DIR = ${YOUR_PATH_TO_THIS_FOLDER_HERE}\20250706\t00z\output `

For example:

` BASE_DATA_DIR=C:\Users\USER\Desktop\Spatiotemporal-Routing-Algorithm\20250706\t00z\outputs `

This is used to indicate where the sample dataset is located

### 3.1 If that dosent work, change line 31 of app.py and replace it with the same path as in your .env


### 4 Then open a terminal in the same folder as the downloaded project and execute this command:

` shiny run --launch-browser app.py `

This will open the app on the address ` http:localhost:8000 ` and will automatically open your browser

## Prerequisites

This was tested only on windows and with python 3.11.4
