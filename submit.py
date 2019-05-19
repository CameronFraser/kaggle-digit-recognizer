import kaggle

kaggle.api.authenticate()

print("Submitting solution")

file_name = ''
submission_description = ''

# kaggle.competition_submit(file_name, submission_description, 'digit_recognizer')