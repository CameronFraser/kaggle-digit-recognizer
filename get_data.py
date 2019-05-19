import kaggle

kaggle.api.authenticate()

kaggle.api.competition_download_files('digit-recognizer', path='./dataset')
