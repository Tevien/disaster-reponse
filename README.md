# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to `http://localhost:3001` in the browser to view the visualisations.

## disaster-reponse files

* [app/](./disaster-reponse/app) - app folder
  * [templates/](./disaster-reponse/app/templates) - HTML templates
    * [go.html](./disaster-reponse/app/templates/go.html) - HTML for text model query page
    * [master.html](./disaster-reponse/app/templates/master.html) - HTML for homepage
  * [run.py](./disaster-reponse/app/run.py) - Main python file for app
* [data/](./disaster-reponse/data)
  * [disaster_categories.csv](./disaster-reponse/data/disaster_categories.csv)
  * [disaster_messages.csv](./disaster-reponse/data/disaster_messages.csv)
  * [process_data.py](./disaster-reponse/data/process_data.py) - ETL pipeline file
* [models/](./disaster-reponse/models)
  * [train_classifier.py](./disaster-reponse/models/train_classifier.py) - Model training pipeline
* [LICENSE](./disaster-reponse/LICENSE)
* [README.md](./disaster-reponse/README.md)
