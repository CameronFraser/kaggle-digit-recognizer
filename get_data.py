"""
    Fetches kaggle dataset.
    Must have kaggle.json file in your .kaggle folder
    for this to work
"""
import kaggle

kaggle.api.authenticate()

print("Downloading digit recognizer data set")
kaggle.api.competition_download_files("digit-recognizer", path="./dataset")
print("Done downloading")
