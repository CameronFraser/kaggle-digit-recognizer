"""
    Submits submission.csv file to kaggle.
    Must have kaggle.json file in your .kaggle folder
    for this to work
"""
import kaggle

kaggle.api.authenticate()

print("Submitting solution")

FILE_NAME = "submission.csv"
DESCRIPTION = (
    "A deep learning approach using CNN's with TF 2.0. "
    "Code: https://github.com/CameronFraser/kaggle-digit-recognizer"
)

kaggle.api.competition_submit(FILE_NAME, DESCRIPTION, "digit-recognizer")

print("Submission complete")
