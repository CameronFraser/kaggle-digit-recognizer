import kaggle

kaggle.api.authenticate()

print("Downloading digit recognizer data set")
kaggle.api.competition_download_files('digit-recognizer', path='./dataset')
